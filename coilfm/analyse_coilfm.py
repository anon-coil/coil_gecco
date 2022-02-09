import numpy as np
import argparse
import importlib
import os, time
import pickle
import matplotlib.pyplot as plt


RESULTS_DIRECTORY = 'results/'
IMAGE_DIRECTORY = 'image/'

image_format = 'pdf' # png

# if the directory to save image does not exist, create it
if not os.path.exists(IMAGE_DIRECTORY):
    os.makedirs(IMAGE_DIRECTORY)


equation_name = 'cdiscon'
 

NUM_RUNS = 100
# variable_list = []

# multi constraint list
variable_constraint_list = ['']

def calculate_distance(num_var, optimal, actual):
    result = 0.0
    for i in range(num_var):
        result += (optimal[i] - actual[i]) ** 2
    return np.sqrt(result)


def get_results(experiment):
    mean_objective_error = []
    mean_constraint_error = []
    stderr_objective_error = []
    stderr_constraint_error = []

    # for variable_constraint in variable_constraint_list:

    equation_name_long = equation_name
    # print(equation_name_long)
    eq = importlib.__import__(equation_name_long) # import equation module
    variable_constraint = '2'
    variable = 2
    data_file = RESULTS_DIRECTORY + equation_name + '_v' + variable_constraint + '_' + experiment + '.pkl'

    optimise_seed, run_results = pickle.load(open(data_file, 'rb'))

    run_results_objective_error = []
    run_results_constraint_error = []

    for run in range(NUM_RUNS):
        # print('variables', run_results[run]['variables'])
        # print(eq.optimal_point)
        # distance_from_optimal_point = calculate_distance(variable, eq.optimal_point, run_results[run]['variables'])
        # print(run_results[run]['distance_from_optimal'])
        # run_results_objective_error.append(distance_from_optimal_point)
        run_results_objective_error.append(run_results[run]['distance_from_optimal']/run_results[run]['optimal']*100.0)
        run_results_constraint_error.append(run_results[run]['distance_from_constraints']/float(variable))

    mean_objective_error.append(np.mean(np.array(run_results_objective_error), axis=0))
    stderr_objective_error.append(np.std(np.array(run_results_objective_error), axis=0)/np.sqrt(NUM_RUNS))

    mean_constraint_error.append(np.mean(np.array(run_results_constraint_error), axis=0))
    stderr_constraint_error.append(np.std(np.array(run_results_constraint_error), axis=0)/np.sqrt(NUM_RUNS))
    return mean_objective_error, stderr_objective_error, mean_constraint_error, stderr_constraint_error


def draw_image(dataset1, dataset1_stderr, dataset1_name, dataset2, dataset2_stderr, dataset2_name, dataset3, dataset3_stderr, dataset3_name, xlabel, ylabel, title, imagename):
    error_kw = dict(lw=0.8, capsize=3, capthick=0.8)
    x = np.arange(len(variable_constraint_list))  # the label locations
    width = 0.2 # the width of the bars
    fig, ax = plt.subplots()
    plt.tick_params(labelsize=14)
    rects1 = ax.bar(x - width, dataset1, width, yerr=dataset1_stderr, error_kw=error_kw, label=dataset1_name, color="#4c72b0")
    rects2 = ax.bar(x, dataset2, width, yerr=dataset2_stderr, error_kw=error_kw, label=dataset2_name, color="#dd8452")
    rects3 = ax.bar(x + width, dataset3, width, yerr=dataset3_stderr, error_kw=error_kw, label=dataset3_name, color="#55a868")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title, fontsize=18)
    # ax.set_xlabel('xxxx', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks([])
    # ax.set_xticks(list(range(len(variable_constraint_list))))
    # ax.set_xticklabels(variable_constraint_list)
    # ax.set_xticks(np.arange(min(x),max(x),1))
    ax.legend(fontsize=14)

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    # plt.show()

    plt.savefig(IMAGE_DIRECTORY + equation_name + '_' + imagename, dpi=72, bbox_inches='tight', pad_inches=0)



stackedcoil_mean_objective_error, stackedcoil_stderr_objective_error, stackedcoil_mean_constraint_error, stackedcoil_stderr_constraint_error = get_results('disconcoil')
vaega_mean_objective_error, vaega_stderr_objective_error, vaega_mean_constraint_error, vaega_stderr_constraint_error = get_results('coil')
ga_mean_objective_error, ga_stderr_objective_error, ga_mean_constraint_error, ga_stderr_constraint_error = get_results('ga')

coil = vaega_mean_objective_error
coil_stderr = vaega_stderr_objective_error
coil_name = 'COIL'
ga = ga_mean_objective_error
ga_stderr = ga_stderr_objective_error
ga_name = 'GA'
stackedcoil = stackedcoil_mean_objective_error
stackedcoil_stderr = stackedcoil_stderr_objective_error
stackedcoil_name = 'COILfm'
xlabel = 'Number of variables'
ylabel = 'Average percentage error'
title = 'Average percentage objective error'
imagename = 'objective_error.' + image_format
draw_image(ga, ga_stderr, ga_name, coil, coil_stderr, coil_name, stackedcoil, stackedcoil_stderr, stackedcoil_name, xlabel, ylabel, title, imagename)

coil = vaega_mean_constraint_error
coil_stderr = vaega_stderr_constraint_error
coil_name = 'COIL'
ga = ga_mean_constraint_error
ga_stderr = ga_stderr_constraint_error
ga_name = 'GA'
stackedcoil = stackedcoil_mean_constraint_error
stackedcoil_stderr = stackedcoil_stderr_constraint_error
stackedcoil_name = 'COILfm'
xlabel = 'Number of variables'
ylabel = 'Average constraint error'
title = 'Average constraint error (per variable)'
imagename = 'constraint_error.' + image_format
draw_image(ga, ga_stderr, ga_name, coil, coil_stderr, coil_name, stackedcoil, stackedcoil_stderr, stackedcoil_name, xlabel, ylabel, title, imagename)
