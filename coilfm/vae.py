from enum import Enum

import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset, Subset

PI = torch.from_numpy(np.asarray(np.pi))


def create_dataloaders(pts, n_train=0.85, n_val=0.5, num_workers=8,
                       batch_size=64):
    N_train = int(0.85 * len(pts))
    N_val = int(0.05 * len(pts))
    N_test = len(pts) - N_train - N_val

    ds = TensorDataset(torch.from_numpy(pts).float())
    rand_indeces = np.random.choice(len(pts), len(pts), replace=False)
    train_inds, val_inds, test_inds = np.split(
        rand_indeces, [N_train, N_train + N_val])
    train_ds = Subset(ds, train_inds)
    val_ds = Subset(ds, val_inds)
    test_ds = Subset(ds, test_inds)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        drop_last=True)
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        drop_last=False)
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        drop_last=False)

    return train_dl, val_dl, test_dl


# -- Prior ---------------------------------------------------------------- -- #
class GaussianMixture(object):
    def __init__(self, mean, std, weights):
        """
        Args:
            mean: Tensor of shape (N, K, d1, ..., dM). Means of the mixtures.
            std: Tensor of shape (N, K, d1, ..., dM). Standard deviation of
                mixtures. Must be same shape as mean.
            weights: Tensor of shape (N, K) or (1, K). Weights of mixtures.
        """
        shape_err_msg = "Mean and std do not have the same shape."
        assert mean.shape == std.shape, shape_err_msg
        weights_dim_err_msg = ("Expected number of weight dimensions to be 2, "
                               "instead got {}".format(weights.dim()))
        assert weights.dim() == 2, weights_dim_err_msg
        shape_err_msg_2 = ("Expected 1st dimension of mean/std to be the same "
                           "as the one of weights.")
        assert mean.shape[1] == weights.shape[1], shape_err_msg_2

        self.mean = mean
        self.std = std
        self.K = mean.shape[1]
        self.weights = weights.view(-1, self.K)
        self.normal = Normal(mean, std)

    def log_prob(self, input):
        """
        Args:
            x: Tensor of shape (N, {1, K}, d1, ..., dM) or
                (L, N, {1, K}, d1, ..., dM).
        Returns
            logp: Tensor of shape (N, {1, K}, 1) or (L, N, {1, K}, 1) similar
                to input shape.
        """
        if len(input.shape) == len(self.mean.shape):
            if self.mean.shape[0] > 1:
                assert input.shape[0] == self.mean.shape[0], \
                    "Input dimension 0 is not the same as mean/std"
            assert input.shape[2:] == self.mean.shape[2:], \
                "Shape error: input.shape[2:] != self.mean.shape[2:]"
            weights = self.weights
        elif len(input.shape) == len(self.mean.shape) + 1:
            weights = self.weights.unsqueeze(0)
        else:
            raise TypeError("Input shape is not compatible")

        log_wx = self.normal.log_prob(input).sum(-1) + torch.log(weights)
        logp = torch.logsumexp(log_wx, -1, keepdim=True)
        return logp

    def sample(self, sample_shape=[1]):
        z_sampler = Normal(self.mean, self.std)
        y_sampler = Categorical(self.weights)
        y = y_sampler.sample(sample_shape=sample_shape)
        z = z_sampler.sample(sample_shape=sample_shape)
        return z[torch.arange(z.shape[0]), 0, y.flatten().long(), :]


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(-1, *self.shape)


class MixturePrior(nn.Module):
    def __init__(self, n_mixture, num_inputs, num_latent, device="cuda:0"):
        super(MixturePrior, self).__init__()
        self.n_mixture = n_mixture

        self.mixture_weights = torch.ones(1, self.n_mixture) / self.n_mixture
        self.mixture_weights = self.mixture_weights.to(device)
        # n_mixture x n_mixture
        self.idle_input = torch.eye(n_mixture, n_mixture, requires_grad=False)
        self.idle_input = self.idle_input.to(device)
        self.idle_encoder = nn.Linear(n_mixture, num_inputs)

        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True))

        self.z_mean = nn.Sequential(
            nn.Linear(256, num_latent),
            Reshape([n_mixture, num_latent]))
        self.z_logvar = nn.Sequential(
            nn.Linear(256, num_latent),
            Reshape([n_mixture, num_latent]),
            nn.Hardtanh(min_val=-6., max_val=0.))

    def forward(self):
        # n_mixture, num_inputs
        h1 = self.idle_encoder(self.idle_input)
        h2 = self.encoder(h1)
        z_mean = self.z_mean(h2)
        z_logvar = self.z_logvar(h2)
        mix_dist = GaussianMixture(
            z_mean,
            torch.exp(0.5 * z_logvar),
            self.mixture_weights
        )

        return mix_dist


class FixedPrior(nn.Module):
    def __init__(self, device="cuda:0", max_val=2., std=0.6):
        super(FixedPrior, self).__init__()
        self.means = torch.Tensor([
            [0., 0.],
            [0., -max_val],
            [0., max_val],
            [max_val, 0.],
            [-max_val, 0.],
            [-max_val, -max_val],
            [max_val, max_val],
            [-max_val, max_val],
            [max_val, -max_val]
        ]).unsqueeze(0).to(device)
        self.std = torch.Tensor(
            [0.6]).view(1, 1).repeat(9, 2).unsqueeze(0).to(device)
        self.mixture_weights = (torch.ones(1, 9) / 9).to(device)

    def forward(self):
        mix_dist = GaussianMixture(
            self.means,
            self.std,
            self.mixture_weights
        )
        return mix_dist


def log_standard_normal(x, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


class FlowPrior(nn.Module):
    def __init__(self, num_latent=2, num_flows=3, num_outputs=2,
                 num_hidden=256, num_hidden_layers=2):
        super(FlowPrior, self).__init__()

        # scale (s) network
        def nets():
            layers = [
                nn.Linear(num_latent // 2, num_hidden),
                nn.LeakyReLU()
            ]
            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(num_hidden, num_hidden))
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(num_hidden, num_latent // 2))
            layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        # translation (t) network
        def nett():
            layers = [
                nn.Linear(num_latent // 2, num_hidden),
                nn.LeakyReLU()
            ]
            for _ in range(num_hidden_layers):
                layers.append(nn.Linear(num_hidden, num_hidden))
                layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(num_hidden, num_latent // 2))
            return nn.Sequential(*layers)

        self.num_outputs = num_outputs

        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        self.num_flows = num_flows

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)

        s = self.s[index](xa)
        t = self.t[index](xa)

        if forward:
            # yb = f^{-1}(x)
            yb = (xb - t) * torch.exp(-s)
        else:
            # xb = f(y)
            yb = torch.exp(s) * xb + t

        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)

        return z, log_det_J

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)

        return x

    def sample(self, batch_size, z=None):
        if z is None:
            z = torch.randn(batch_size, self.num_outputs)
        x = self.f_inv(z)
        return x.view(-1, self.num_outputs)

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        log_p = (log_standard_normal(z) + log_det_J.unsqueeze(1))
        return log_p

# -- Architecture --------------------------------------------------------- -- #


class EncoderMLP(nn.Module):
    def __init__(self, num_inputs=2, num_hidden=300, num_latent=2,
                 num_layers=1):
        super(EncoderMLP, self).__init__()
        layers = [
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.ReLU(inplace=True))
        self.encode = nn.Sequential(*layers)
        self.mean = nn.Linear(num_hidden, num_latent)
        self.logvar = nn.Sequential(
            nn.Linear(num_hidden, num_latent),
            nn.Hardtanh(min_val=-6, max_val=1.))

    def forward(self, input):
        h = self.encode(input)
        mean = self.mean(h)
        logvar = self.logvar(h)
        return mean, logvar


class DecoderMLP(nn.Module):
    def __init__(self, num_inputs=2, num_latent=2, num_hidden=300,
                 lik="gaussian", num_layers=1):
        super(DecoderMLP, self).__init__()
        layers = [
            nn.Linear(num_latent, num_hidden),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.ReLU(inplace=True))
        self.decode = nn.Sequential(*layers)
        if lik == "gaussian":
            self.x_mean = nn.Sequential(
                nn.Linear(num_hidden, num_inputs)
            )
            self.x_logvar = nn.Sequential(
                nn.Linear(num_hidden, num_inputs),
                nn.Hardtanh(min_val=-6, max_val=-2)
            )
        elif lik == "mse":
            self.x = nn.Linear(num_hidden, num_inputs)
        self.lik = lik

    def forward(self, input):
        h = self.decode(input)
        if self.lik == "mse":
            rec = self.x(h)
        elif self.lik == "gaussian":
            x_mean = self.x_mean(h)
            x_logvar = self.x_logvar(h)
            rec = torch.cat((x_mean, x_logvar), -1)
        return rec


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    PI = torch.from_numpy(np.asarray(np.pi))
    log_p = -0.5 * torch.log(
        2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


class VAE(nn.Module):
    def __init__(self, num_inputs=2, num_latent=2, num_hidden=256, lik="mse",
                 prior="gaussian", num_mixture=9, num_flows=3,
                 num_flow_layers=2, num_autoencoding_layers=1):
        super(VAE, self).__init__()
        self.encoder = EncoderMLP(
            num_inputs=num_inputs, num_latent=num_latent,
            num_hidden=num_hidden, num_layers=num_autoencoding_layers)
        self.decoder = DecoderMLP(
            num_inputs=num_inputs, num_latent=num_latent,
            num_hidden=num_hidden, lik=lik, num_layers=num_autoencoding_layers)
        self.lik = lik
        self.prior = prior
        if self.prior == "mixture":
            self.pz = MixturePrior(num_mixture, num_inputs, num_latent)
        elif self.prior == "fixedmixture":
            self.pz = FixedPrior()
        elif self.prior == "flow":
            self.pz = FlowPrior(num_outputs=num_latent,
                                num_latent=num_latent,
                                num_hidden=num_hidden, num_flows=num_flows,
                                num_hidden_layers=num_flow_layers)

    def forward(self, input):
        mean, logvar = self.encoder(input)
        z = self.reparameterize(mean, logvar)
        rec = self.decoder(z)
        return rec, mean, logvar, z

    def loss(self, x, rec, mu, logvar, z, kl_weight=1.0):
        if self.lik == "mse":
            lik = F.mse_loss(rec, x, reduction='sum') / x.shape[0]
        elif self.lik == "gaussian":
            latent_dim = rec.shape[1] // 2
            mu_x, logvar_x = torch.split(rec, [latent_dim, latent_dim], dim=1)
            lik = - log_normal_diag(x, mu_x, logvar_x).sum() / x.shape[0]
        else:
            raise NotImplementedError
        if self.prior == "gaussian":
            KL = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        elif self.prior in ["mixture", "fixedmixture"]:
            qz = Normal(mu, torch.exp(0.5 * logvar))
            log_pz = self.pz().log_prob(z.unsqueeze(1)).view(z.shape[0], 1)
            log_qz = qz.log_prob(z).view(z.shape[0], -1)
            KL = (log_qz.sum(-1, keepdim=True) - log_pz.sum(
                -1, keepdim=True)).mean()
        elif self.prior == "flow":
            KL = (log_normal_diag(
                z, mu, logvar) - self.pz.log_prob(z)).sum() / mu.shape[0]
        else:
            raise NotImplementedError
        loss = lik + (kl_weight * KL)
        stats = {
            "loss": loss.detach().item(),
            "kl": KL.detach().item(),
            "lik": lik.detach().item()
        }
        return loss, stats

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def save(self, path):
        torch.save(self.state_dict(), path)

    def express(self, z):
        # Use VAE as a generator, given numpy latent vec return full numpy
        # phenotype
        if self.prior == "flow":
            latent = torch.from_numpy(z).float()
            latent = self.pz.sample(batch_size=latent.shape[0], z=latent)
        else:
            latent = torch.from_numpy(z).float()
        pheno = self.decoder(latent)
        return pheno.detach().numpy()


# -- Utils ---------------------------------------------------------------- -- #
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        return fmtstr.format(**self.__dict__)


# -- Training ------------------------------------------------------------- -- #
def train_vae(model, train_dl, val_dl,
              batch_size=100, lr=5e-4, epochs=100, beta=1.,
              print_freq=100, notebook_display=False):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        losses = AverageMeter('Loss', ':.4f')
        kls = AverageMeter('KL', ':.4f')
        liks = AverageMeter('Lik', ':.4f')
        progress = ProgressMeter(
            len(train_dl),
            [losses, kls, liks],
            prefix="Epoch: [{}]".format(epoch))
        train_samples = []
        train_rec = []
        for i, feats in enumerate(train_dl):
            feats = feats[0].to("cuda:0")
            rec, mu, logvar, z = model(feats)
            loss, stats = model.loss(
                feats, rec, mu, logvar, z, kl_weight=min(epoch / 10, beta))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(stats["loss"], feats.shape[0])
            kls.update(stats["kl"], feats.shape[0])
            liks.update(stats["lik"], feats.shape[0])
            if i % print_freq == 0:
                progress.display(i)
            train_samples.append(feats.cpu().numpy())
            train_rec.append(rec.detach().cpu().numpy())
        train_samples = np.concatenate(train_samples, 0)
        train_rec = np.concatenate(train_rec, 0)

        progress.display_summary()

        val_samples = []
        val_rec = []
        with torch.no_grad():
            for i, feats in enumerate(val_dl):
                feats = feats[0].to("cuda:0")
                rec, _, _, _ = model(feats)
                val_samples.append(feats.cpu().numpy())
                val_rec.append(rec.detach().cpu().numpy())
        val_samples = np.concatenate(val_samples, 0)
        val_rec = np.concatenate(val_rec, 0)

        if notebook_display:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.scatter(
                train_samples[::100, 0], train_samples[::100, 1], c="r", label="data")
            ax1.scatter(train_rec[::100, 0],
                        train_rec[::100, 1], c="b", label="rec")
            ax2.scatter(val_samples[::100, 0], val_samples[::100, 1], c="r",
                        label="data")
            ax2.scatter(val_rec[::100, 0],
                        val_rec[::100, 1], c="b", label="rec")
            plt.show()


def sample_2d_pts(res=15):
    x = np.linspace(-3, 3, res)
    y = np.linspace(-3, 3, res)
    X, Y = np.meshgrid(x, y)  # grid of point
    pts = np.c_[X.flatten(), Y.flatten()]
    return pts


if __name__ == "__main__":
    # ---- Code Testing ---- #
    # - Load Data
    from dejong import create_dataset
    dataset = create_dataset()
    train_dl, val_dl, test_dl = create_dataloaders(dataset)

    # - Setup and Train VAE
    model_file = 'dejong_test.pt'
    model = VAE(prior="fixedmixture").to("cuda:0")  # use constructor defaults

    # train or load
    # train_vae(model, train_dl, val_dl)
    # torch.save(model.state_dict(), model_file)
    model.load_state_dict(torch.load(model_file))
    model.to('cpu')

    # - 'Express' from genotype
    pts = sample_2d_pts(15)
    model.express(pts)

    print("Done")
