from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Sequential
import keras
import numpy as np
import Auxiliary.funcs_filtered_outputs_m_3 as m3
from cmath import inf
import Auxiliary.s_box as sbox
import os
from statistics import mean, pstdev
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout = open('BayesianOptimizationResults_4Rules_2Layer.txt', 'w')

NUM_RULES = 4
INPUT_SIZE = 8
OUTPUT_SIZE = 8
MAX_STATES = 256

# This function converts the integer list representation to a version that can be given as input to the ANN
# Parameters:
#   state: The list of function index in the list rules_list. Values range from 0-55.
def convert(state):
    converted = np.zeros((NUM_RULES*56), dtype=np.int8)
    for i, j in enumerate(state):
        converted[i*56+j] = 1
    return converted

# This function does the inverse of convert(). It converts the ANN input to a form easily understandable.
# Parameters:
#   state: Input to the ANN.
def convert_8(state):
    r = np.where(state == 1)[0]
    res = [s % 56 for s in r]
    return res

rules_list, rule_names = m3.return_rules()

# Creating the function approximator (ANN)
if NUM_RULES == 4:
    try:
        model = keras.models.load_model('ann4')
        print('Importing Model..')
    except:
        model = Sequential()
        model.add(Dense(56*NUM_RULES, input_shape=(56*NUM_RULES,), batch_size=1, name='inputlayer'))
        model.add(Dense(56, activation='relu', name='hiddenlayer'))
        model.add(Dense(1, activation='relu', name='outputlayer'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Bayesian Optimization
exceptions = []
visited_states = []
strengths = []
DUs = []
NLs = []
best_strength = -inf
best_DU = -inf
best_NL = -inf
best_states = []

# Define the search space for the Bayesian Optimization
space = [Integer(0, 55, name=f"x{i}") for i in range(NUM_RULES)]

# Objective function to minimize (negative strength, since we want to maximize strength)
@use_named_args(space)
def objective(**params):
    S = [params[f"x{i}"] for i in range(NUM_RULES)]
    S_converted = convert(S)
    try:
        if S in exceptions:
            return 1e6  # Large penalty for invalid states
        R, DU, NL = sbox.state_crypto_strength(S, True)
        if NL == 0:
            exceptions.append(S)
            return 1e6  # Large penalty for linear states
        visited_states.append(S)
        strengths.append(R)
        DUs.append(DU)
        NLs.append(NL)
        global best_strength, best_DU, best_NL, best_states
        if R > best_strength:
            best_strength = R
            best_DU = DU
            best_NL = NL
            best_states = [S]
        elif R == best_strength and S not in best_states:
            best_states.append(S)
        return -R
    except Exception as e:
        print(f"Exception: {e}")
        return 1e6  # Large penalty for any exceptions

# Run Bayesian Optimization
result = gp_minimize(objective, space, n_calls=MAX_STATES, random_state=0)

print("Exiting the Bayesian Optimization...\n")
print("FINAL REPORT:")
print(f'Number of States Visited: {len(visited_states)}')
print(f"Best Strength: {round(best_strength, 2)}")
print(f'Best DU: {best_DU} Best NL: {best_NL}')
print(f'Average Strength of visited States: {round(mean(strengths), 2)}')
print(f'Standard Deviation of strength: {round(pstdev(strengths), 2)}')
print(f'Number of states with best strengths: {len(best_states)}')
print(f'No. of linear States: {len(exceptions)}')
print('---------------------------------------------------------------------')
print('Best States:')
print(best_states)
print('Exceptions: ')
print(exceptions)
sys.stdout.close()
sys.stdout = sys.__stdout__
