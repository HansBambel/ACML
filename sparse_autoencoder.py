import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivedSigmoid(x):
    return sigmoid(x)/(1-sigmoid(x))

def forwardPass(x):
    # add bias to samples
    inputToHidden = np.dot(np.append([1], x), weightsIH)
    a2 = sigmoid(inputToHidden)
    inputToOutput = np.dot(np.append([1], a2), weightsHO)
    output = sigmoid(inputToOutput)
    return output

examples = np.eye(8)
# examples = [[1, 0, 0, 0, 0, 0, 0, 0]]
y = examples
weightsIH = np.random.rand(9, 3)
weightsHO = np.random.rand(4, 8)
# weightsIH = np.zeros((9, 3))
# weightsHO = np.zeros((4, 8))
# print(f'weightsIH {weightsIH}')
# print(forwardPass([[1, 0, 0, 0, 0, 0, 0, 0]]))

batchSize= 4
iterations = 0
ALPHA = 0.01
GAMMA = 0.7
converged = False
while not converged:
    totalError = 0
    samples = examples[np.random.randint(len(examples), size=batchSize)]
    # samples = np.random.permutation(examples)
    # print(samples)
    y = samples
    iterations += 1
    # forward pass --> backprop
    # 8 nodes+bias X 3 nodes+bias X 8 nodes
    #### Forward pass ####
    inputBias = np.concatenate((np.ones(len(samples)).reshape(-1, 1), samples), axis=1)
    hiddenInput = np.dot(inputBias, weightsIH)
    # print(f"hiddenInput.shape. Should be (numSamples, 3) {hiddenInput.shape}")
    activationHidden = sigmoid(hiddenInput)
    outputHiddenBias = np.concatenate((np.ones(len(samples)).reshape(-1, 1), activationHidden), axis=1)
    # print(f"outputHiddenBias.shape. Should be (numSamples, 4) {outputHiddenBias.shape}")

    inputLastLayer = np.dot(outputHiddenBias, weightsHO)
    yPred = sigmoid(inputLastLayer)
    # print(f"yPred: {yPred}")
    # print(f'y: {y}')

    #### Backprop now ####
    outputError = yPred - y
    # print("outputError")
    # print(outputError)
    # print(outputError.shape)
    deltaOutput = outputError * derivedSigmoid(yPred)
    # print(np.sum(outputError, axis=1))
    totalError = np.sum(outputError)#, axis=1)

    errorHidden = np.dot(deltaOutput, weightsHO.T)
    deltaHidden = errorHidden[:, 1:] * derivedSigmoid(activationHidden)

    # # adapt the biases
    weightsIH[0, :] -= 1/len(samples)*ALPHA*np.dot(inputBias[:, 0].T, deltaHidden)
    weightsHO[0, :] -= 1/len(samples)*ALPHA*np.dot(outputHiddenBias[:, 0].T, deltaOutput)
    # adapt the weights
    weightsIH[1:, :] -= 1/len(samples)*ALPHA*(np.dot(inputBias[:, 1:].T, deltaHidden) + GAMMA*weightsIH[1:, :])
    weightsHO[1:, :] -= 1/len(samples)*ALPHA*(np.dot(outputHiddenBias[:, 1:].T, deltaOutput) + GAMMA*weightsHO[1:, :])


    # weightsIH += ALPHA*np.dot(inputBias.T, deltaHidden)
    # weightsHO += ALPHA*np.dot(outputHiddenBias.T, deltaOutput)
    # TODO regression
    # weightsIH += 1/len(examples)*(GAMMA*(weightsIH.T-ALPHA*(delta1.T)))

    print(totalError)
    # if iterations > 10:
    #     break
    if np.abs(totalError) <= 0.00001:
        converged = True
# print(delta1)
print(f'Converged after {iterations} iterations')

print('Testing:')
for sample in examples:
    print(f'Input: {sample}')
    print(f'Output: {forwardPass(sample)}')
