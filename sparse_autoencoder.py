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
    print(f'hidden layer activation: {np.around(a2, 3)}')
    return output


examples = np.eye(8)
y = examples
weightsIH = np.random.rand(9, 3)
weightsHO = np.random.rand(4, 8)
# weightsIH = np.zeros((9, 3))
# weightsHO = np.zeros((4, 8))
# print(f'weightsIH {weightsIH}')

batchSize = 4
ALPHA = 0.1
LAMBDA = 0.001

epochs = 0
totalError = 0
converged = False
while not converged:
# while epochs < 150000:
    totalError = 0
    # samples = examples[np.random.randint(len(examples), size=batchSize)]
    samples = np.random.permutation(examples)
    y = samples
    epochs += 1
    # forward pass --> backprop
    # 8 nodes+bias X 3 nodes+bias X 8 nodes
    updateWeightsIH = np.zeros((9, 3))
    updateWeightsHO = np.zeros((4, 8))
    for sample in samples:
        ### Forwardpass ###
        activationInput = np.append([1], sample).reshape(-1, 1)
        inputToHidden = np.dot(activationInput.T, weightsIH)
        a2 = sigmoid(inputToHidden)
        activationHidden = np.append([1], a2).reshape(-1, 1)
        inputToOutput = np.dot(activationHidden.T, weightsHO)
        yPred = sigmoid(inputToOutput)
        # print(f'activationHidden {activationHidden.shape}')

        ### Backpropagation ###
        # Error of Output
        delta3 = yPred - sample
        delta2 = np.multiply(np.dot(delta3, weightsHO.T), derivedSigmoid(activationHidden.T))
        # print(f'delta2 {delta2.shape}')
        updateWeightsHO += np.dot(activationHidden, delta3)
        updateWeightsIH += np.dot(activationInput, delta2[:, 1:])
        # The total error should actually be the sum of the absolute errors, but
        # this leads to no convergence ---> error somewhere else
        totalError += np.sum(delta3)

    # print(f'updateWeightsIH: {updateWeightsIH}')
    # print(f'updateWeightsHO: {updateWeightsHO}')
    m = len(samples)  # number of samples
    # dividing by the number of samples makes it slower to converge
    # update the bias weights
    weightsHO[0, :] -= ALPHA/m*updateWeightsHO[0, :]
    weightsIH[0, :] -= ALPHA/m*updateWeightsIH[0, :]
    # update the other weights
    weightsHO[1:, :] -= ALPHA*(updateWeightsHO[1:, :]/m + LAMBDA*weightsHO[1:, :])
    weightsIH[1:, :] -= ALPHA*(updateWeightsIH[1:, :]/m + LAMBDA*weightsIH[1:, :])

    # if epochs%100==0:
    print(f'Error after {epochs} epochs: {totalError}')

    if np.abs(totalError) <= 0.0001:
        converged = True
print(f'Converged after {epochs} epochs with error {totalError}')


print('Testing:')
for sample in examples:
    print(f'Input: {sample}')
    print(f'Output: {np.around(forwardPass(sample), 3)}')

print(f'weightsIH: \n {weightsIH}')
print(f'weightsHO: \n {weightsHO}')
