"""
Crawl through a folder containing path estimates for a particular section, and
a particular meas. noise ensemble member, and order the paths by action level.
"""

import numpy as np
import sys

# get args from command line
#L, s, M = sys.argv[1:]
#L = int(L)  # no. of measured input/output neurons
#M = int(M)  # no. of trainining examples
M, DH = sys.argv[1:]
M = int(M)  # number of training examples
DH = int(DH)  # hidden layer size

# annealing parameters
Ninit = 50
Nbeta = 436
#Nbeta = 195
Nens = 1
#secID = 2

for i in range(Nens):
    print(i+1)

    # specify folder containing path estimates
    #result_dir = "cluster_runs/N10_D10_sigmoid_trainingN100/L%d_%s_%dex/"%(L, s, Nex)
    #result_dir = "local_annealing_runs/N10_D10_Lvarious/L10_sm0p05/"
    #result_dir = "cluster_runs/barcent_N3_DH30/%dex/"%(Nex)
    result_dir = "../boom/DH%d_%dex/"%(DH, M)

    # first, get all of the action values
    action_vals = np.zeros((Ninit, Nbeta))

    init_missing = False
    for j in range(Ninit):
        try:
            action_vals[j, :] = np.load(result_dir + "action_errors_%d.npy"%(j+1,))[:, 1]
        except:
            print(result_dir)
            print("File " + result_dir + "action_errors_%d.npy"%(j+1,) + " missing")
            init_missing = True
            break

    # now sort the action values, and store along with the initial path indices
    if init_missing:
        break
    else:
        sorted_indices = np.argsort(action_vals, axis=0).T
        np.savetxt(result_dir + "sorted_indices.dat", sorted_indices, fmt="%d")
