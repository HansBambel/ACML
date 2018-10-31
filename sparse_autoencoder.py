import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


examples = np.eye(8)
x = np.vstack((np.ones(len(examples)), examples))
y = np.arange(8)
weightsIH = np.random.rand(9, 3)
weightsHO = np.random.rand(4, 8)

ALPHA = 0.1
GAMMA = 0.7
# while not converged:
# forward pass --> backprop
# 8 nodes+bias X 3 nodes+bias X 8 nodes
# print(np.sum(np.dot(x.T, weightsIH), axis=1))
hiddenActivation = np.dot(x.T, weightsIH)
print(hiddenActivation)
hidden = sigmoid(hiddenActivation[0, 0])


print(hidden)
# output = np.dot
