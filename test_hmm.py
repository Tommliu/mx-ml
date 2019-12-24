#from sklearn import datasets
#from numpy import loadtxt
#from line_profiler import LineProfiler
#from collections import Counter
import mxnet as mx
from mxnet.test_utils import set_default_context
from mxnet import npx
import time
import csv

# Using Offcial Numpy to Generate training data.
def HMM_Generate(N, L): 
    import numpy as np
    import numpy_ml as ml

    states = np.array([0, 1, 2])
    n_states = len(states)
    observations = np.array([0, 1])
    n_observations = len(observations)
    start_probability = np.array([0.2, 0.4, 0.4])
    transition_probability = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])
    emission_probability = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])

    hmm1 = ml.hmm.MultinomialHMM(transition_probability, emission_probability, start_probability)
    train_data = []
    latent_states_data = []
    for i in range(N):
        latent_states, _emissions = hmm1.generate(n_steps=L, latent_state_types=states, obs_types=observations)
        train_data.append(_emissions.flatten())
        latent_states_data.append(latent_states.flatten())
    test_emission = np.atleast_2d(np.random.randint(2, size=L))
    np.savetxt("MultinomialHMM_Train_Emission_{}.csv".format(N), np.array(train_data, dtype=int), fmt='%d', delimiter=",")
    np.savetxt("MultinomialHMM_Train_Latent_{}.csv".format(N), np.array(latent_states_data, dtype=int), fmt='%d', delimiter=",")
    np.savetxt("MultinomialHMM_Test_Emission_{}.csv".format(N),test_emission ,delimiter=",", fmt='%i')

# test Hidden Markov models
def test_hmm(op_type, trails, states, observations, _emissions, test_emission):
    time_start= time.time()
    for i in range(trails):
        hmm2 = ml.hmm.MultinomialHMM() # A=None, B=None, pi=None, eps=None
        hmm2.fit(_emissions, states, observations, tol=0.01, verbose=True) # max_iter=100, tol=0.001, verbose=False
        if op_type == 'DeepNumPy CPU':
            mx.nd.waitall()
    time_end = time.time()    
    test_log_p = hmm2.log_likelihood(test_emission.reshape(1, -1))
    test_latent_state = hmm2.decode(test_emission)
    print("----------------------")
    print(trails, "trails:", op_type, "in dataset of shape", _emissions.shape, "consumed: ", time_end - time_start, " seconds")
    print("Loglikekihood: ", test_log_p)
    print("Probability: ", np.exp(test_log_p))
    return np.exp(test_log_p)

#op_types = ['Official Numpy', 'DeepNumPy CPU', 'DeepNumPy GPU']
op_types = ['Official Numpy'] 
trails = 1
N = 200
L = 10
#HMM_Generate(N, L)

for op_type in op_types:
    if op_type == 'Official Numpy':
        import numpy as np
        import numpy_ml as ml
    elif op_type == 'DeepNumPy CPU':
        from mxnet import numpy as np
        import deepnumpy_ml as ml
        set_default_context(mx.cpu(0))
        npx.set_np()   
    else:
        from mxnet import numpy as np
        import deepnumpy_ml as ml
        set_default_context(mx.gpu(0))
        npx.set_np()

    states = np.array([0, 1, 2])
    observations = np.array([0, 1])
    
    with open("MultinomialHMM_Train_Emission_{}.csv".format(N), newline='') as csvfile:
        train_emissions = np.array(list((csv.reader(csvfile))),dtype=int)
    with open("MultinomialHMM_Test_Emission_{}.csv".format(N), newline='') as csvfile:
        test_emission = np.array(list(csv.reader(csvfile)),dtype=int).reshape(1, -1)
    p = test_hmm(op_type, trails, states, observations, train_emissions, test_emission)
