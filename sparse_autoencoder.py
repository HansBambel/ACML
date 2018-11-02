import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivedSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def forwardPass(x):
    # add bias to samples
    inputToHidden = np.dot(np.append([1], x), weightsIH)
    a2 = sigmoid(inputToHidden)
    inputToOutput = np.dot(np.append([1], a2), weightsHO)
    output = sigmoid(inputToOutput)
    return output


examples = np.eye(8)
y = examples
weightsIH = np.random.rand(9, 3)
weightsHO = np.random.rand(4, 8)
# weightsIH = np.zeros((9, 3))
# weightsHO = np.zeros((4, 8))
# print(f'weightsIH {weightsIH}')
# print(forwardPass([[1, 0, 0, 0, 0, 0, 0, 0]]))

batchSize = 4
iterations = 0
ALPHA = 0.1
LAMBDA = 0.3
converged = False
while not converged:
# while iterations < 100000:
    totalError = 0
    # samples = examples[np.random.randint(len(examples), size=batchSize)]
    samples = np.random.permutation(examples)
    # samples = [[1, 0, 0, 0, 0, 0, 0, 0]]
    # print(samples)
    y = samples
    iterations += 1
    # forward pass --> backprop
    # 8 nodes+bias X 3 nodes+bias X 8 nodes
    updateWeightsIH = np.zeros((9, 3))
    updateWeightsHO = np.zeros((4, 8))
    for sample in samples:
        # Forwardpass
        activationInput = np.append([1], sample).reshape(-1, 1)
        inputToHidden = np.dot(activationInput.T, weightsIH)
        a2 = sigmoid(inputToHidden)
        activationHidden = np.append([1], a2).reshape(-1, 1)
        inputToOutput = np.dot(activationHidden.T, weightsHO)
        yPred = sigmoid(inputToOutput)
        # Backpropagation
        delta3 = yPred - sample
        # print(f'delta3 {delta3.shape}')
        # print(f'activationHidden {activationHidden.shape}')
        # delta2 = np.dot(weightsHO, delta3.T) * derivedSigmoid(activationHidden)
        delta2 = np.multiply(np.dot(delta3, weightsHO.T), derivedSigmoid(activationHidden.T))
        # print(f'delta2 {delta2.shape}')
        updateWeightsHO += np.dot(activationHidden, delta3)
        updateWeightsIH += np.dot(activationInput, delta2[:, 1:])
        totalError += np.sum(delta3)
    # print(f'updateWeightsIH: {updateWeightsIH}')
    # print(f'updateWeightsHO: {updateWeightsHO}')
    # break
    m = len(samples)  # number of samples
    # update the bias weights
    weightsHO[0, :] -= ALPHA*updateWeightsHO[0, :]
    weightsIH[0, :] -= ALPHA*updateWeightsIH[0, :]
    # update the other weights
    weightsHO[1:, :] -= ALPHA*updateWeightsHO[1:, :]
    weightsIH[1:, :] -= ALPHA*updateWeightsIH[1:, :]
    # weightsHO[1:, :] = (1 - ALPHA*LAMBDA/m)*weightsHO[1:, : ] - ALPHA*
    # weightsIH[1:, :] = ALPHA/m * (updateWeightsIH[1:, :] + LAMBDA * weightsIH[1:, :])
    # weightsHO[1:, :] = ALPHA/m * (updateWeightsHO[1:, :] + LAMBDA * weightsHO[1:, :])
    # weightsIH[1:, :] = ALPHA/m * (updateWeightsIH[1:, :] + LAMBDA * weightsIH[1:, :])

    print(totalError)
    # if iterations > 10:
    #     break
    if np.abs(totalError) <= 0.01:
        converged = True
# print(delta1)
print(f'Converged after {iterations} iterations')

print('Testing:')
for sample in examples:
    print(f'Input: {sample}')
    print(f'Output: {forwardPass(sample)}')
