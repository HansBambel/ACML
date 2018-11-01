import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivedSigmoid(x):
    return sigmoid(x)/(1-sigmoid(x))


examples = np.eye(8)
examples = [[1, 0, 0, 0, 0, 0, 0, 0]]
y = examples
weightsIH = np.random.rand(9, 3)
weightsHO = np.random.rand(4, 8)
# weightsIH = np.zeros((9, 3))
# weightsHO = np.zeros((4, 8))

ALPHA = 0.01
GAMMA = 0.7
# while not converged:
# forward pass --> backprop
totalError = 0
# 8 nodes+bias X 3 nodes+bias X 8 nodes
inputBias = np.concatenate((np.ones(len(examples)).reshape(-1, 1), examples), axis=1)
hiddenInput = np.dot(inputBias, weightsIH)
print(f"hiddenInput.shape. Should be (numSamples, 3) {hiddenInput.shape}")
activationHidden = sigmoid(hiddenInput)
outputHiddenBias = np.concatenate((np.ones(len(examples)).reshape(-1, 1), activationHidden), axis=1)
print(f"outputHiddenBias.shape. Should be (numSamples, 4) {outputHiddenBias.shape}")

yPred = np.dot(outputHiddenBias, weightsHO)
yPred = sigmoid(yPred)
print("yPred")
print(yPred)

# Backprop now
outputError = yPred - y
totalError += outputError
print("outputError")
print(outputError)
print(outputError.shape)
deltaOutput = outputError* derivedSigmoid(yPred)
errorHidden = np.dot(deltaOutput, weightsHO.T)
deltaHidden = errorHidden[:, 1:] * derivedSigmoid(activationHidden)

# delta2 = np.dot(outputError, weightsHO.T) * derivedSigmoid(outputHiddenBias)
print(f"deltaOutput: {deltaOutput}")
# delta1 = np.dot(delta2[:, 1:], weightsIH.T) * derivedSigmoid(inputBias)
print(f"deltaHidden: {deltaHidden}")
print(np.dot(yPred.T, deltaOutput).shape)
weightsHO += ALPHA*np.dot(yPred.T, deltaOutput)
weightsIH += ALPHA*np.dot(activationHidden.T, deltaHidden)
# weightsIH += 1/len(examples)*(GAMMA*(weightsIH.T-ALPHA*(delta1.T)))

# print(delta1)

# TODO regularization
