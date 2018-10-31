import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivedSigmoid(x):
    return sigmoid(x)/(1-sigmoid(x))


examples = np.eye(8)
# examples = [[1, 0, 0, 0, 0, 0, 0, 0]]
examplesBias = np.concatenate((np.ones(len(examples)).reshape(-1, 1), examples), axis=1)
y = examples
weightsIH = np.random.rand(9, 3)
weightsHO = np.random.rand(4, 8)

ALPHA = 0.1
GAMMA = 0.7
# while not converged:
# forward pass --> backprop
# 8 nodes+bias X 3 nodes+bias X 8 nodes
hiddenActivation = np.dot(examplesBias, weightsIH)
outputHidden = sigmoid(hiddenActivation)

# print(np.ones(len(examples)))
outputHiddenBias = np.vstack([np.ones(len(examples)), outputHidden.T])
yPred = np.dot(outputHiddenBias.T, weightsHO)
yPred = sigmoid(yPred)

# print(yPred)
# Backprop now
deltaOutput = yPred - y
# print(deltaOutput)
# print(np.dot(outputHiddenBias, deltaOutput))
deltaHidden = np.dot(np.dot(outputHiddenBias, deltaOutput), derivedSigmoid(yPred))

print(deltaHidden)

# TODO regularization
