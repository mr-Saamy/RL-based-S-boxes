# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools
from matplotlib.pyplot import figure
import Auxiliary.funcs_filtered_outputs_m_3 as m3
import Auxiliary.funcs_filtered_outputs_m_4 as m4
import Auxiliary.funcs_filtered_outputs_m_5 as m5
import Auxiliary.funcs_filtered_outputs_m_6 as m6

#Helper Functions

# Convert a decimal number into a corresponding array of bit values
def decimalToBinary(n): 
    return "{0:09b}".format(int(n))
  
# Convert the binary representation array into the corresponding decimal value
def BinaryTodecimal(bit_list):
  dec=0
  for bit in bit_list:
    dec=(dec<<1)|bit
  return dec

# Function which runs the CA rule over the CA and xors the rule outputs and returns a single bit value 
# Parameters:
# rule - the list of CA rules to be run on the CA
# nb_size - Size of the neighbourhood considered
# ca_len  - Number of cells in the Cellular Automata
# ca  - The input CA configuration to run the rules on

def rule_op(rule,nb_size,ca_len,ca):
  ops=[]
  for i in range(0,ca_len,nb_size):
    ops.append(rule(ca[i:i+nb_size+1]))
  res=0
  for op in ops:
    res^=op
  return res

#CA Rules Definitions which we run on the Cellular Automata. These are rules of neighbourhood size 3
rule_list,rule_names= m3.return_rules()

rules=[m3.rule_210,m3.rule_108,m3.rule_68,m3.rule_238,m3.rule_120,m3.rule_180,m3.rule_30,m3.rule_92]
# rules=rule_list[:8]

# Function to emulate the output of the S-box. It outputs the inputs and corresponding outputs in a bit list form.
# Parameters:
# rules - The list of CA rules which we will run.

def Sbox(rules):
  inputs=[]
  outputs=[]
  for i in range(256):
    res = list(map(int, str(decimalToBinary(i))))
    
    inputs.append(res[1:])
    op=[]
    for rule in rules:
      op.append(rule_op(rule,3,8,res))
    outputs.append(op)
    # print(f'{res[1:]} ->{op}')
  return inputs,outputs

inputs,outputs=Sbox(rules)
# We convert the bit list of outputs into decimal values
decimal_repr=[]
for bit_list in outputs:
  decimal_repr.append(BinaryTodecimal(bit_list))

figure(figsize=(10,6))
plt.hist(decimal_repr,bins=256);

# Function to check the bijectivity of the s-box function. Returns 1 if bijective, 0 if not
# Parameters:
# decimal_repr - Decimal Representation of the outputs produced by S-box
def bijectivity(decimal_repr):
  len_ops=len(decimal_repr)
  len_distinct=len(set(decimal_repr))
  if (len_ops==len_distinct):
    print("It is Bijective")
    return 1
  else:
    print("Not Bijective")
    return 0

bijectivity(decimal_repr)

# Function to Calculate the difference Distribution Table of the S-box function and returns its differential uniformity
# Parameters:
# decimal_repr - Decimal Representation of the outputs produced by S-box
def diff_uniformity(decimal_repr):
  ddt=np.zeros((256,256))
  for a in range(256):
    for x in range(256):
      sum=x^a
      F1=decimal_repr[sum]
      F2=decimal_repr[x]
      b=F1^F2
      ddt[a][b]+=1
  for i in range(256):
    ddt[i][i]=0
  ddt[0]=np.zeros(256)
  return (np.amax(ddt))

diff_uniformity(decimal_repr)