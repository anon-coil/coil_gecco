import importlib
import random
import numpy as np
import pickle
import time
import argparse

from matplotlib import pyplot as plt

# for VAE
import torch
# from learn_representation import VecVAE
from vae import VAE

VAE_DIRECTORY = 'vae/'
DATA_DIRECTORY = 'data/'

EXPRESSED_WITH_VAE = True
RANDOM_POINT = False # either random point or an existing data point it learnt from
USE_SET_DATA = True

#----------
# Get command line arguments
#----------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--equation', help='equation name should correspond to the equation file, e.g., enter eq01 to import the equation file eq01.py')
parser.add_argument('-v', '--num_variables', help='enter the number of variables')
parser.add_argument('-r', '--runs', help='enter the number of runs. if the argument is not provided, it defaults to 1 run')
parser.add_argument('-s', '--seed', help='enter -1 for a random seed, or enter the seed number. if the argument is not provided, it defaults to -1.')
parser.add_argument('-d', '--debug', action='store_true', help='if argument is used, generate debug info.')


args = parser.parse_args()

#----------
# Import equation module
#----------
if args.equation:
    equation_name = args.equation
    eq = importlib.__import__(equation_name) # import equation module
else:
    exit("Error. Please specify equation name in the command line. Use --help for more information.")

#----------
# Get number of variables
#----------
if args.num_variables:
    NUM_VARIABLES = int(args.num_variables)
    if NUM_VARIABLES < eq.MIN_VARIABLES:
        exit("Error. Minimum number of variables for this function is %d. Use --help for more information." % eq.MIN_VARIABLES)
else:
    NUM_VARIABLES = eq.MIN_VARIABLES

#----------
# Set seed
#----------
if not args.seed or int(args.seed) < 0: # if args.seed is not provided or is negative
    seed = int(time.time()) # use current time as random seed
else:
    seed = int(args.seed)
print('Seed', seed)
random.seed(seed)

#----------
# Load vae and scaler
#----------
vae_flow_file = VAE_DIRECTORY + 'flow_vae_dejong5-largest.pt'

lik = "mse"
num_latent = 2
num_hidden = 512
batch_size = 100
lr = 5e-4
epochs = 100
beta = 1.

vae = VAE(num_inputs=2, num_latent=num_latent, num_hidden=num_hidden, lik=lik, prior="flow", num_flows=5, num_flow_layers=4, num_autoencoding_layers=2)

vae_f_state_dict = torch.load(vae_flow_file, map_location=torch.device('cpu'))
vae.load_state_dict(vae_f_state_dict["model_state_dict"])

scaler_file = VAE_DIRECTORY + equation_name + '_v' + str(NUM_VARIABLES) + '_vae_scaler.pkl'
learn_representation_seed, scaler = pickle.load(open(scaler_file, 'rb'))



def read_individual_data_file(data_file):
    data = pickle.load(open(data_file, 'rb'))
    return data

def check_near(x, num):
    num_min = num-0.1
    num_max = num+0.1
    if x <= num_max and x >= num_min:
        return True
    else:
        return False

def check_bucket(x, bucket_low, bucket_high):
    if x <= bucket_high and x > bucket_low:
        return True
    else:
        return False

def run_tests():

    list_of_points = []

    buckets = [
    [-50, -40],
    [-40, -30],
    [-30, -20],
    [-20, -10],
    [-10, 0],
    [0, 10],
    [10, 20],
    [20, 30],
    [30, 40],
    [40, 50]
    ]
    # buckets = [
    # [-50, -30],
    # [-30, -10],
    # [-10, 10],
    # [10, 30],
    # [30, 50],
    # ]
    bucket_count = [0] * len(buckets)
    for i in range(100):
        random_genome = np.random.uniform(eq.VAEGA_MIN_RANGE, eq.VAEGA_MAX_RANGE, NUM_VARIABLES)
        sampled_point = vae.express(np.asarray([random_genome]))
        print('random genome', random_genome)
        print('sampled point', sampled_point[0])
        data_point = sampled_point[0]*50
        list_of_points.append(data_point)
        if check_near(data_point[0], 0.0) and check_near(data_point[1], -10.0):# and check_near(data_point[2], 0.0) and check_near(data_point[3], 0.0):
            print(data_point)

        for index, bucket in enumerate(buckets):
            bucket_low = bucket[0]
            bucket_high = bucket[1]
            if check_bucket(data_point[0], bucket_low, bucket_high) and check_bucket(data_point[1], bucket_low, bucket_high) and check_bucket(data_point[2], bucket_low, bucket_high) and check_bucket(data_point[3], bucket_low, bucket_high):
                bucket_count[index] += 1
                break

    for i in range(len(buckets)):
        print(str(buckets[i]), str(bucket_count[i]))

    # for item in list_of_points:
    #     data_point = ''
    #     for i in item:
    #         data_point += str(i*50) + '\t' # quick way to make it -50 50
    #     print(data_point)

    # for i in range(eq.NUM_CONSTRAINTS):
    #     constraint_id = 'g' + str(i+1)
    #     data = read_individual_data_file(DATA_DIRECTORY + '/data_' + constraint_id + '.pkl')
    #     raw_data = np.array(data)
    #     g_diff = []
    #     for i in raw_data:
    #         if RANDOM_POINT:
    #             reconstructed_point = vae.reconstruct(np.random.uniform(low=-1.0, high=1.0, size=IND_SIZE))
    #         else:
    #             reconstructed_point = vae.reconstruct(i)
    #         g_diff.append(abs(reconstructed_point - i))

    #     np_g_diff = np.array(g_diff)
    #     print(constraint_id + ',' + str(np.mean(np_g_diff)) + ',' + str(np.std(np_g_diff)) + ',' + str(len(g_diff)))


    # for i in range(eq.NUM_CONSTRAINTS):
    #     constraint_id = 'g' + str(i+1)
    #     # data = read_individual_data_file(DATA_DIRECTORY + '/data_' + constraint_id + '.pkl')
    #     # print(data)
    #     data_range = np.arange(-2.0, 2.0, 0.1)
    #     data = []
    #     for item in data_range:
    #         data.append([0.0, item])
    #     raw_data = np.array(data)
    #     g_diff = []
    #     # for i in range(10000):
    #     for i in raw_data:
    #         if USE_SET_DATA:
    #             random_point = np.array([i]) # np.random.uniform(low=-1.0, high=1.0, size=NUM_LATENT)
    #         else:
    #             random_point = np.random.uniform(low=-1.0, high=1.0, size=(1, NUM_VARIABLES))
    #         # random_point = np.random.uniform(low=-1.0, high=1.0, size=NUM_LATENT)
    #         # print('random point', random_point)
    #         if EXPRESSED_WITH_VAE:
    #             # print('random point', random_point)
    #             # print('vae expressing')
    #             raw_expressed_point = vae.express(random_point)
    #             # print(raw_expressed_point)
    #             # print('scaler inverse transform')
    #             # print(scaler)
    #             expressed_point = scaler.inverse_transform(raw_expressed_point)
    #             # print(expressed_point)
    #             # print('expressed point', expressed_point)
    #             unnormalised_item = unnormalised(expressed_point[0])
    #         else:
    #             unnormalised_item = unnormalised(random_point)

    #         if USE_SET_DATA:
    #             print('unnormalised_item\t', unnormalised_item[0])
    #         distance_from_constraint = globals()['calculate_' + constraint_id](unnormalised_item)
    #         # print('distance_from constraint', distance_from_constraint)
    #         if (distance_from_constraint > 0.0):
    #             g_diff.append(distance_from_constraint)


    #     if (len(g_diff) > 0):
    #         np_g_diff = np.array(g_diff)
    #         print(constraint_id + ',' + str(np.mean(np_g_diff)) + ',' + str(np.std(np_g_diff)) + ', constraint broken: ' + str(len(g_diff)))
    #     else:
    #         print(constraint_id + ',' + '0' + ',' + 'N/A' + ', constraint broken: ' + '0')




if __name__ == "__main__":
    run_tests()

