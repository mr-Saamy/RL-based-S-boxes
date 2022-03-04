# from scipy.fft import set_backend
from cmath import inf
from cv2 import threshold
import Auxiliary.s_box as sbox
import os
from random import random,randint,choice
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import Auxiliary.funcs_filtered_outputs_m_3 as m3
import numpy as np
from keras.models import Sequential
import keras
from keras.layers import Dense
rules_list,rule_names=m3.return_rules()

# rules=[m3.rule_178,m3.rule_92,m3.rule_154,m3.rule_18,m3.rule_68,m3.rule_172,m3.rule_222,m3.rule_46]
# def loss_function(R_bar,R):
#     def loss(y_true,y_pred):
        
    
    
def convert(state):
    converted=np.zeros((8,56),dtype=np.int8)
    for i,j in enumerate(state):
        converted[i][j]=1
    return converted

def convert_8(state):
    res=[]
    for s in state:
        res.append(s.index(1))
    return res

model = Sequential()
model.add(Dense(56, input_shape=(8,56), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()
average_reward=0
beta=0.1
epsilon_threshold=0.1
alpha=0.05
gamma=.8
# rules=[m3.rule_178,m3.rule_92,m3.rule_154,m3.rule_18,m3.rule_68,m3.rule_172,m3.rule_222,m3.rule_46]
S=[29, 53, 42, 36, 20, 52, 6, 16]
A=(randint(0,7),randint(0,55))

while True:
    print(f'Current State: {S}')
    S_prime=S.copy()
    S_prime[A[0]]=A[1]
    R=sbox.state_crypto_strength(S_prime)
    epsilon = random()
    if epsilon<epsilon_threshold:
        A_prime=(randint(0,7),randint(0,55))
        taken_state=S_prime.copy()
        taken_state[A_prime[0]]=A_prime[1]
        taken_state=convert(taken_state)
    else:
        max_states=[]
        max_actions=[]
        max_value=-inf
        for i in range(8):
            for j in range(56):
                if j in S_prime:
                    continue
                next_state=np.zeros((8,56),dtype=np.int8)
                for k in S_prime:
                    if k==S_prime[i]:
                        continue
                    next_state[i][k]=1
                next_state[i][j]=1
                print('input')
                print(np.shape(np.array([next_state])))
                prediction=model.predict(np.array([next_state]))
                print("prediction")
                print(prediction)
                if prediction>max_value:
                    max_value=prediction
                    max_states=[next_state]
                    max_actions=[(i,j)]
                elif prediction==max_value:
                    max_states.append(next_state)
                    max_actions.append((i,j))
        action_index=randint(0,len(max_actions)-1)
        taken_state=max_states[action_index]
        A_prime=max_actions[action_index]
    print("State")
    print(taken_state)
    Q_s_a=model.predict(np.array([convert(S_prime)]))+sbox.state_crypto_strength(S_prime)
    Q_sprime_aprime=model.predict(np.array(taken_state))+sbox.state_crypto_strength(convert_8(taken_state))
    target=Q_s_a+alpha*(R-average_reward+gamma*(Q_sprime_aprime)-Q_s_a)
    print(np.array([target]))
    print(np.array([convert(S_prime)]))
    model.fit(np.array([convert(S_prime)])
                ,np.array([target]))
    delta=R-average_reward+Q_sprime_aprime-Q_s_a
    average_reward+=beta*delta
    S=S_prime
    A=A_prime
                
                
                    
                
    
    
