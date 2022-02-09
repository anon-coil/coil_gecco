import numpy as np

# generate data: specify number of data points generated
NUM_DATA_POINTS = 5000

# the minimum number of variables required for the equation to work
MIN_VARIABLES = 2

MAX_VARIABLES = 2

# specify objective function
def func(x, num_var):
    result = 0.0
    for i in range(num_var):
        result += x[i]**2
    return result

# specify whether to minimise or maximise function, 0 for min 1 for max
MIN_OR_MAX_FLAG = 0

# set the min and max range for the variables
X_MIN_RANGE = -50.0
X_MAX_RANGE = 50.0


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

# specify constraints (return 0 if constraint is met, otherwise return absolute distance)
def c0(x, num_var): # dejong_constraint
    threshold = 445

    # Scale
    zoom = .39
    _x1 = x[0]*zoom
    _x2 = x[1]*zoom
    
    # Rotate
    theta = np.radians(-33)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    _x1, _x2 = R@np.array([_x1, _x2])
    
    # Translate
    _x1 += 9
    _x2 += -3    
    
    result = dejong5([_x1], [_x2])[0]
    # print(result)
    # Evaluate
    if result < threshold:
        return 0.0
    else:
        return np.abs(result-threshold)

# list of constraints: add specified constraints to this list in order for them to be considered
CONSTRAINTS = [
c0,
]

# calculate the optimal result for the function for the constraint(s) to be met
def calculate_optimal(num_var):
    optimal_point = [5.9, 5.28535]
    return func(optimal_point, num_var)


# generate data: specify num gen and num pop for the data generator GA
DATAGEN_GEN = 200
DATAGEN_POP = 200

# # generate data: specify min and max range for data
DATAGEN_MIN_RANGE = -1.0
DATAGEN_MAX_RANGE = 1.0

# learn representation: specify the number of latent variables and epochs for the vae
# NUM_LATENT = NUM_VARIABLES
NUM_EPOCHS = 200

# optimise: specify num gen and num pop for the optimiser GA
VAEGA_GEN = 50
VAEGA_POP = 20

# optimse: the range for the GA to generate random numbers for the latent variable
VAEGA_MIN_RANGE = -2.0
VAEGA_MAX_RANGE = 2.0

# comparison GA: specify num gen and num pop for the GA
# GA_NUM_INDIVIDUALS = NUM_VARIABLES # the number of individuals for the GA is the number of variables
GA_GEN = 50
GA_POP = 20


