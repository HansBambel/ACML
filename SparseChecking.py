import numpy as np
import matplotlib.pyplot as plt


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

weightsIH = np.random.rand(9, 3)
weightsHO = np.random.rand(4, 8)
LAMBDA = 0.001

stored = np.zeros((6, 2))  # Stores number epochs and total error
alphaValues = [2, 1.5, 1, 0.8, 0.5, 0.2]
colors = ['yellow', 'blue', 'red', 'black', 'orange', 'green']
errors = [[] for i in alphaValues]

#alphaValues = [0.8]
for i,value in enumerate(alphaValues):
    print (f'Alpha: {value}')
    print('***')

    weightsIH = np.random.rand(9, 3)
    weightsHO = np.random.rand(4, 8)

    ALPHA = value

    converged = False
    epochs = 0

    #for i in range(0,500):
        #print ('This is i ')
        #print(i)
    while not converged:
    # while epochs < 150000:
        totalError = 0
        # samples = examples[np.random.randint(len(examples), size=batchSize)]
        samples = np.random.permutation(examples)
        # samples = examples
        # samples = [[1, 0, 0, 0, 0, 0, 0, 0]]
        # print(samples)
        y = samples
        #print(samples)
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
        #print(f'Error after {epochs} epochs: {totalError}') #Only printing every 50 (below)

        if np.abs(totalError) <= 0.005:
            converged = True

        # if epochs%25 == 0:
        errors[i].append(totalError)
        #     if epochs <= 55:
        #         plt.plot(epochs, totalError, 'o', color=colors[i], label=f'Alpha = {alphaValues[i]}')
        #
        #     plt.plot(epochs, totalError, 'o', color=colors[i])
        #     print(f'Error after {epochs} epochs: {totalError}') #Only printing every 50

    print(f'Converged after {epochs} epochs with error {totalError}')
    stored[i,0] = epochs
    stored[i,1] = totalError


    print('Testing:')
    for sample in examples:
        print(f'Input: {sample}')
        print(f'Output: {np.around(forwardPass(sample), 3)}')

    print(f'weightsIH: \n {np.around(weightsIH, 1)}')
    print(f'weightsHO: \n {np.around(weightsHO, 1)}')

print(' ')
print (f'Stored: \n {np.around(stored, 5)}')
# plt.axis([0, 30000, -0.5, 1])
# print(errors)
for i, e in enumerate(errors):
    plt.plot(e, color=colors[i], label=f'Alpha = {alphaValues[i]}')
plt.ylim(0, 0.8)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.show()
