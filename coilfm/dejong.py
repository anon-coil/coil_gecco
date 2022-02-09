import numpy as np

import matplotlib.pyplot as plt


# Benchmark function
def dejong5(X1, X2):
    n_vals = len(X1)
    Y = np.full( (n_vals), np.nan)
    for i in range(n_vals):    
        x1, x2 = X1[i], X2[i]
        total = 0

        A = np.zeros((2,25))
        a = np.array([-32, -16, 0, 16, 32])

        A[0,:] = np.tile(a,(1,5))
        A[1,:] = np.sort(np.tile(a,(1,5)))

        for ii in range(25):
            a1i = A[0,ii]
            a2i = A[1,ii]
            term1 = ii+1
            term2 = (x1 - a1i) ** 6
            term3 = (x2 - a2i) ** 6
            new = 1 / (term1 + term2 + term3)
            total += new        
        Y[i] = y = 1 / (0.002 + total)
    return Y

# As a constraint only
def dejong_constraint(X1,X2, threshold=445):
    """  1) Expected Range of [-50 to 50]
         2) Optima (0,0) in invalid region    
    """    
    # Scale
    zoom = .39
    _x1 = X1*zoom
    _x2 = X2*zoom
    
    # Rotate
    theta = np.radians(-33)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    _x1, _x2 = R@np.array([_x1, _x2])
    
    # Translate
    _x1 += 9
    _x2 += -3    
    
    # Evaluate
    valid = (dejong5(_x1, _x2) < threshold)    
    return valid
    

def create_dataset(random=False, res=100, n_samples=1000):
    # Collect dataset of valid solutions
    abs_range = 50
    if random:
        x = (2*(np.random.rand(n_samples)-0.5)) * abs_range
        y = (2*(np.random.rand(n_samples)-0.5)) * abs_range
        Z = dejong_constraint(x,y)
        pts = np.c_[x,y]
    else:
        x = np.linspace(-abs_range, abs_range, res)
        y = np.linspace(-abs_range, abs_range, res)
        X, Y = np.meshgrid(x, y) # grid of point
        Z = dejong_constraint(X.flatten(), Y.flatten())
        pts = np.c_[X.flatten(),Y.flatten()]
        
    valid_pts = pts[Z,:]    
    return valid_pts

def main():
    "Saves valid dataset as 'dejong_dataset.csv' "
    n_samples = 5000
    valid_pts = create_dataset(random=True)
    training_set = [valid_pts]

    while len(np.vstack(training_set)) < n_samples:
        valid_pts = create_dataset(random=True)
        training_set += [valid_pts]
    training_set = np.vstack(training_set)
    training_set = training_set[:n_samples]
    np.savetxt('dejong_dataset.csv', training_set)

    # Visualize
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(training_set[:,0], training_set[:,1], s=1)
    ax.scatter(0,0,c='r',s=80)
    ax.set(xlim=[-51,51], ylim=[-51,51])
    fig.savefig('dejong_valid_solutions.png')

    print(f'[*] Done: {len(training_set)} data points created')


    ## -- Train Classic VAE -- #
    print('\n[*] Training VAE')
    # Taken from vae_datagen
    from vae_basic import VecVAE, train_vae
    from sklearn import preprocessing

    raw_data = training_set
    scaler = preprocessing.StandardScaler().fit(raw_data) # zero mean unit standard deviation
    genomes = scaler.transform(raw_data)

    n_dim, n_latent, n_epochs = genomes.shape[1], 2, 1000

    vae = VecVAE(n_dim, n_latent)
    vae = train_vae(genomes, vae, n_epochs, view_mod=25)
    vae.save('dejong_vae.pt')






if __name__ == "__main__":
    main()

