
# coding: utf-8

# In[ ]:


"""
Example deep neural network annealing.
"""

import numpy as np
from varanneal import va_nnet
import sys, time

ninit = 15#int(sys.argv[1])
M = 100#int(sys.argv[2])
D_hidden = 250#int(sys.argv[3])
adolcID = 0#int(sys.argv[4])

# Define the transfer function
def sigmoid(x, W, b):
    linpart = np.dot(W, x) + b
    return 1.0 / (1.0 + np.exp(-linpart))

# Network structure
N = 3  # Total number of layers
D_in = 200  # Number of neurons in the input layer
D_out = 100  # Number of neurons in the output layer
#D_hidden =   # Number of neurons in the hidden layers

structure = np.zeros(N, dtype='int')
structure[0] = D_in  # 3 neurons in the input layer
structure[N-1] = D_out  # 2 neurons in the output layer
for i in range(1, N-1):
    structure[i] = D_hidden  # 5 neurons in the hidden layers

Lidx = [np.linspace(0, D_in-1, D_in, dtype='int'), np.linspace(0, D_out-1, D_out, dtype='int')]

################################################################################
# Action/annealing parameters
################################################################################
# RM, RF0
RM = 1.0
RF0 = 1.0e-8 * RM * float(np.sum(structure) - structure[0]) / float(structure[0] + structure[-1])
# alpha, and beta ladder
alpha = 1.1
beta_array = np.linspace(0, 435, 436)
epochs = 5
batch_size = 20
################################################################################
# Input and output data
################################################################################

print(np.load('generated_l96_data/l96_001/gen_l96_noisy_0_01.npy'))# has 5 l96 variables

data  = np.load('generated_l96_data/l96_001/gen_l96_noisy_0_01.npy')
data = data[0:M+D_in+D_out]
Didx = 0

# normalize
#data = data / np.max(np.abs(data))

#add noise
#noise = np.random.normal(scale=0.02, size=data.shape[0])
#data[:, Didx] += noise

T = data.shape[1]

data_in = np.zeros((M, D_in))
data_out = np.zeros((M, D_out))

for i in xrange(M):
    data_in[i] = data[i:i+D_in, Didx]
    data_out[i] = data[i+D_in:i+D_in+D_out, Didx]




#train_examples = 10000
#x_train = x_all[0:train_examples,:]
#y_train = y_all[0:train_examples,:]

################################################################################
# Initial path/parameter guesses
################################################################################
DHmax = 1000
ninitmax = 100
np.random.seed(27509436 + (M-1)*D_in*DHmax*ninitmax + D_hidden*ninit)
# Neuron states
Xin = np.random.randn(D_in)
Xin = (Xin - np.average(Xin)) / np.std(Xin)
#X0 = [Xin]
X0 = np.copy(Xin)
for n in xrange(N-2):
    X0 = np.append(X0, 0.2*np.random.rand(D_hidden) + 0.4)
X0 = np.append(X0, 0.2*np.random.rand(D_out) + 0.4)

for m in xrange(M - 1):
    Xin = np.random.randn(D_in)
    Xin = (Xin - np.average(Xin)) / np.std(Xin)
    X0 = np.append(X0, Xin)
    for n in xrange(N-2):
        X0 = np.append(X0, 0.2*np.random.rand(D_hidden) + 0.4)
    X0 = np.append(X0, 0.2*np.random.rand(D_out) + 0.4)

X0 = np.array(X0).flatten()

# Parameters
NP = np.sum(structure[1:]*structure[:-1] + structure[1:])
#Pidx = []
P0 = np.array([], dtype=np.float64)

W_i0 = 0
W_if = structure[0]*structure[1]
b_i0 = W_if
b_if = b_i0 + structure[1]

for n in xrange(N - 1):
    if n == 0:
        Pidx = np.arange(W_i0, W_if, 1, dtype='int')
    else:
        Pidx = np.append(Pidx, np.arange(W_i0, W_if, 1, dtype='int'))
    if n == 0:
        P0 = np.append(P0, (2.0*np.random.rand(structure[n]*structure[n+1]) - 1.0) / D_in)
    else:
        P0 = np.append(P0, (2.0*np.random.rand(structure[n]*structure[n+1]) - 1.0) / D_hidden)
    P0 = np.append(P0, np.zeros(structure[n+1]))

    if n < N - 2:
        W_i0 = b_if
        W_if = W_i0 + structure[n+1]*structure[n+2]
        b_i0 = W_if
        b_if = b_i0 + structure[n+2]

P0 = np.array(P0).flatten()
Pidx = np.array(Pidx).flatten().tolist()

################################################################################
# Annealing
################################################################################
# Initialize Annealer
anneal1 = va_nnet.Annealer()
# Set the network structure
anneal1.set_structure(structure)
# Set the activation function
anneal1.set_activation(sigmoid)
# Set the input and output data
anneal1.set_input_data(data_in)
anneal1.set_output_data(data_out)

# Run the annealing using L-BFGS-B
BFGS_options = {'gtol':1.0e-12, 'ftol':1.0e-10, 'maxfun':1000000, 'maxiter':1000000}
tstart = time.time()
anneal1.anneal(X0, P0, alpha, beta_array, RM, RF0, Pidx, Lidx=Lidx,
               method='L-BFGS-B', opt_args=BFGS_options, adolcID=adolcID)
print("\nADOL-C annealing completed in %f s."%(time.time() - tstart))

# Save the results of annealing
#anneal1.save_states("L%d_%s_%dex/states_%d.npy"%(L, suffix, M, ninit))
#anneal1.save_params("params.npy")
#anneal1.save_paths("l96_out_data/DH%d_%dex/io_%d.npy"%(D_hidden, M, ninit))
anneal1.save_io("l96_out_data/DH%d_%dex/io_%d.npy"%(D_hidden, M, ninit))
anneal1.save_Wb("l96_out_data/DH%d_%dex/W_%d.npy"%(D_hidden, M, ninit),
                "l96_out_data/DH%d_%dex/b_%d.npy"%(D_hidden, M, ninit))
anneal1.save_action_errors("l96_out_data/DH%d_%dex/action_errors_%d.npy"%(D_hidden, M, ninit))