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
# examples = [[1, 0, 0, 0, 0, 0, 0, 0]]
y = examples
weightsIH = np.random.rand(9, 3)
weightsHO = np.random.rand(4, 8)
# weightsIH = np.zeros((9, 3))
# weightsHO = np.zeros((4, 8))
# print(f'weightsIH {weightsIH}')
print(forwardPass([[1, 0, 0, 0, 0, 0, 0, 0]]))

batchSize = 4
iterations = 0
ALPHA = 0.01
GAMMA = 0.7
converged = False
while iterations < 1000:
    totalError = 0
    # samples = examples[np.random.randint(len(examples), size=batchSize)]
    samples = np.random.permutation(examples)
    # print(samples)
    y = samples
    iterations += 1
    # forward pass --> backprop
    # 8 nodes+bias X 3 nodes+bias X 8 nodes
    updateWeightsIH = np.zeros((9, 3))
    updateWeightsHO = np.zeros((4, 8))
    for sample in samples:
        inputToHidden = np.dot(np.append([1], sample), weightsIH)
        a2 = sigmoid(inputToHidden)
        inputToOutput = np.dot(np.append([1], a2), weightsHO)
        yPred = sigmoid(inputToOutput)
        # TODO backprop
        delta3 = yPred - sample
        totalError += np.sum(delta3)
        delta2 = np.dot(weightsHO, delta3) * derivedSigmoid(np.append([1], a2))
        updateWeightsHO += np.dot(np.append([1], a2).reshape(-1, 1), delta3.reshape(-1, 1).T)
        # print(delta2.shape)
        updateWeightsIH += np.dot(np.append([1], sample).reshape(-1, 1), delta2.reshape(-1, 1)[1:].T)
        # delta2 = np.dot(np.dot(weightsHO, error).reshape(-1, 1), derivedSigmoid(yPred).reshape(-1, 1).T)
        # delta1 = np.dot(np.dot(weightsIH, delta2).reshape(-1, 1), derivedSigmoid(a2).reshape(-1, 1).T)
    # update the bias weights
    # print(updateWeightsIH)
    # print(updateWeightsHO)
    m = 3
    weightsHO[0, :] = 1/m * updateWeightsHO[0, :]
    weightsIH[0, :] = 1/m * updateWeightsIH[0, :]
    # update the other weights
    weightsHO[1:, :] = 1/m * (updateWeightsHO[1:, :] + GAMMA * weightsHO[1:, :])
    weightsIH[1:, :] = 1/m * (updateWeightsIH[1:, :] + GAMMA * weightsIH[1:, :])


    #### Forward pass ####
    # inputBias = np.concatenate((np.ones(len(samples)).reshape(-1, 1), samples), axis=1)
    # hiddenInput = np.dot(inputBias, weightsIH)
    # # print(f"hiddenInput.shape. Should be (numSamples, 3) {hiddenInput.shape}")
    # activationHidden = sigmoid(hiddenInput)
    # outputHiddenBias = np.concatenate((np.ones(len(samples)).reshape(-1, 1), activationHidden), axis=1)
    # # print(f"outputHiddenBias.shape. Should be (numSamples, 4) {outputHiddenBias.shape}")
    #
    # inputLastLayer = np.dot(outputHiddenBias, weightsHO)
    # yPred = sigmoid(inputLastLayer)
    # # print(f"yPred: {yPred}")
    # # print(f'y: {y}')
    #
    # #### Backprop now ####
    # outputError = yPred - y
    # # print("outputError")
    # # print(outputError)
    # # print(outputError.shape)
    # deltaOutput = outputError * derivedSigmoid(yPred)
    # # print(deltaOutput)
    # # print(np.sum(outputError, axis=1))
    # totalError = np.sum(outputError)#, axis=1)
    #
    # errorHidden = np.dot(deltaOutput, weightsHO.T)
    # # print(errorHidden)
    # deltaHidden = errorHidden[:, 1:] * derivedSigmoid(activationHidden)
    #
    # # # adapt the biases
    # weightsIH[0, :] -= 1/len(samples)*ALPHA*np.dot(inputBias[:, 0].T, deltaHidden)
    # weightsHO[0, :] -= 1/len(samples)*ALPHA*np.dot(outputHiddenBias[:, 0].T, deltaOutput)
    # # adapt the weights
    # weightsIH[1:, :] -= 1/len(samples)*ALPHA*(np.dot(inputBias[:, 1:].T, deltaHidden) + GAMMA*weightsIH[1:, :])
    # weightsHO[1:, :] -= 1/len(samples)*ALPHA*(np.dot(outputHiddenBias[:, 1:].T, deltaOutput) + GAMMA*weightsHO[1:, :])

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
