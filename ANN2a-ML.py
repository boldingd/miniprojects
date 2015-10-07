import numpy as np
import numpy.random as npr

import random

# weight decay is making it *much worse*
# wat

class Dataset:
  def __init__(self):
    pass

  def get_sample(self):
    s = npr.uniform(low = -1.0, high = 1.0, size = [3])
    l = np.zeros([1])

#    if s[0] > -0.2 and s[1] > -0.2: # non-linear rule
#      l[0] = 1.0
#    else:
#      l[0] = -1.0

    rad = s[0] ** 2
    rad += s[1] ** 2
    rad += s[2] ** 2
    rad = rad ** 0.5

    if rad > 0.5:
      l[0] = -1.0
    else:
      l[0] = 1.0

    return (s, l)

d = Dataset()
dtrain = [ d.get_sample() for _ in range(500) ]
dtest = [ d.get_sample() for _ in range(200) ]

def sigmoid(x):
  return np.tanh(x)

def d_sigmoid(x):
  return 1.0 - (np.tanh(x) ** 2)

ins = 3
hiddens = 4
outs = 1

ih_weights = npr.uniform(low = -0.2, high = 0.2, size=[hiddens, ins+1]) # plus a bias unit
ho_weights = npr.uniform(low = -2.0, high = 2.0, size=[outs, hiddens])

learn_rate = 0.8

ih_last_delta = None
ho_last_delta = None
momentum = 0.2

for i in range(1000):
  for (sample, labels) in dtrain:  
    
    biased_sample = np.ones([ins+1])
    for i in range(len(sample)): # gotta be a better way to do this
      biased_sample[i] = sample[i]
    # biased_sample[i] = 1.0, for biase unit

    ih_total_activations = np.dot(ih_weights, biased_sample)
    ih_outs = sigmoid(ih_total_activations)
    ho_total_activations = np.dot(ho_weights, ih_outs)
    ho_outs = sigmoid(ho_total_activations)
  
    errors = ho_outs - labels # d/dx(error)

    # backprop term
    bp_errors = np.dot(ho_weights.T, errors * d_sigmoid(ho_total_activations))
  
    ho_deltas = errors * d_sigmoid(ho_total_activations)
    ho_deltas = np.dot(ho_deltas.reshape([outs,1]), ih_outs.reshape([1,hiddens])) # reshape into column and row vectors
    ho_deltas *= learn_rate
    
    ho_weights -= ho_deltas
  
    # momentum term
    if ho_last_delta is not None:
      ho_weights -= (momentum * ho_last_delta)
    ho_last_delta = ho_deltas
    
    # update ih
    ih_deltas = bp_errors * d_sigmoid(ih_total_activations)
    ih_deltas = np.dot(ih_deltas.reshape([hiddens,1]), biased_sample.reshape([1,ins+1])) # reshape into column and row vectors
    ih_deltas *= learn_rate
    
    ih_weights -= ih_deltas
  
    # momentum term
    if ih_last_delta is not None:
      ih_weights -= (momentum * ih_last_delta)
    ih_last_delta = ih_deltas

  # weight decay
  ho_weights *= 0.99
  ih_weights *= 0.99

    # inspect a few runs, after we get into the training
#    if i % 100 == 0 and i > 0:
#      if random.uniform(0.0, 1.0) < 0.05:
#        print(str(ih_last_delta))

right = 0
wrong = 0

for (sample, labels) in dtest:
  biased_sample = np.ones([ins+1])
  for i in range(ins):
    biased_sample[i] = sample[i]

  ih_total_activations = np.dot(ih_weights, biased_sample)
  ih_outs = sigmoid(ih_total_activations)
  ho_total_activations = np.dot(ho_weights, ih_outs)
  ho_outs = sigmoid(ho_total_activations)

  #print(str(sample) + " " + str(labels) + "  ->  " + str(ho_outs))

  dif = labels - ho_outs
  dif = dif[0]
  if dif < 0:
    dif *= -1.0
  if dif > 0.1:
    wrong += 1
    print("wrong:  " + str(labels) + "  ->  " + str(ho_outs))
    rad = sample.dot(sample) ** 0.5
    print("r = " + str(rad))
  else:
    right += 1

print("right:  " + str(right))
print("wrong:  " + str(wrong))

