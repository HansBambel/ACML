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
y = examples

weightsIH = np.random.rand(9, 3)
weightsHO = np.random.rand(4, 8)

#weightsIH = np.zeros((9, 3))
#weightsHO = np.zeros((4, 8))
# print(f'weightsIH {weightsIH}')
# print(forwardPass([[1, 0, 0, 0, 0, 0, 0, 0]]))

batchSize = 4
ALPHA = 0.1
#LAMBDA = 0.0001
LAMBDA = 0.000

epochs = 0
totalError = 0
converged = False

#while not converged:


stored = np.zeros((5, 2))


alphaValues = [1,0.5, 0.1, 0.05, 0.01]
# alphaValues = [5]
for i,value in enumerate(alphaValues):

    print (i)
    print (value)
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
        #samples = np.random.permutation(examples)
        samples = examples
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
            #print(yPred)

            ### Backpropagation ###
            # Error of Output
            delta3 = yPred - sample
            #print('This is delta 3')
            #print(delta3)

            delta2 = np.multiply(np.dot(delta3, weightsHO.T), derivedSigmoid(activationHidden.T))
            #print('This is delta 2')
            #print(delta2)
            # print(f'delta2 {delta2.shape}')
            #print(activationHidden)
            #print(delta3)
            updateWeightsHO += np.dot(activationHidden, delta3)
            #print('This is updateWeight')
            #print(updateWeightsHO)
            updateWeightsIH += np.dot(activationInput, delta2[:, 1:])
            totalError += np.sum(delta3)
            #print('LoopOver')

        # print(f'updateWeightsIH: {updateWeightsIH}')
        # print(f'updateWeightsHO: {updateWeightsHO}')
        m = len(samples)  # number of samples
        # dividing by the number of samples makes it slower to converge
        # update the bias weights
        weightsHO[0, :] -= ALPHA/m*updateWeightsHO[0, :]
        weightsIH[0, :] -= ALPHA/m*updateWeightsIH[0, :]
        # update the other weights
        weightsHO[1:, :] -= ALPHA*((updateWeightsHO[1:, :])/m + LAMBDA*weightsHO[1:, :])
        weightsIH[1:, :] -= ALPHA*((updateWeightsIH[1:, :])/m + LAMBDA*weightsIH[1:, :])

        # if epochs%100==0:
        print(f'Error after {epochs} epochs: {totalError}')

        if np.abs(totalError) <= 0.001:
            converged = True

        if epochs%500 == 0:
            plt.plot(epochs, totalError, 'ro')

    print(f'Converged after {epochs} epochs with error {totalError}')
    stored[i,0] = epochs
    stored[i,1] = totalError


    print('Testing:')
    for sample in examples:
        print(f'Input: {sample}')
        print(f'Output: {np.around(forwardPass(sample), 3)}')

    print(f'weightsIH: \n {weightsIH}')
    print(f'weightsHO: \n {weightsHO}')

print (stored)
plt.axis([0, 3000, -5, 5])
plt.show()
