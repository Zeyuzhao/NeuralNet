import random

import numpy as np
import mnist_loader

class NeuralNet():
    VALUE_SHIFT = -2
    def __init__(self, dataSet, dim = []):
        '''
        dataSet: a list of tuples, (X, y) describing the input and corresponding output
        dim: a list of the networks dimensions: [2, 4, 3] would be 2 inputs, 4 neuron layer, 3 output layer
        activations: first layer would be input, last layer would be output, starts from a(2), a(l)
        z: list of z values, starts from z(1) to z(l), z(1) is
        weights: list of weights, starts from w(2) and ends at w(l)
        '''
        self.hiddenLayersDim = dim[1:len(dim)]
        self.totalLayerDim = dim
        self.numLayers = len(dim)
        self.weights = []
        self.initWeights()

        self.z = [np.zeros((b,1)) for b in self.hiddenLayersDim]
        self.bias = [np.random.random(b) for b in self.hiddenLayersDim]
        self.activations = [np.zeros((b, 1)) for b in self.hiddenLayersDim]

        self.dataSet = list(dataSet)
        self.shuffleList()


    def fowardProp(self, dataNum):
        x = self.xSet[dataNum]
        z2 = np.dot(self.getWeights(2), x) + self.getBias(2)
        self.setZ(2, z2)
        a2 = self.sigmoid(z2)
        self.setActivations(2, a2)
        for i in range(3, self.numLayers):
            #z and b from layer 2 to l
            #a from layer 1 to l
            currentZ = np.dot(self.getWeights(i), self.getActivations(i - 1)) + self.getBias(i)
            self.setZ(i, currentZ)
            currentA = self.sigmoid(currentZ)
            self.setActivations(i, currentA)
        return self.getActivations(self.numLayers)

    def computeCost(self, dataNum):
        yHat = self.fowardProp(dataNum)
        y = self.ySet[dataNum]
        return np.linalg.norm(y - yHat) ** 2 / 2

    def backProp(self, dataNum):
        yHat = self.fowardProp(dataNum)
        y = self.ySet[dataNum]
        diff = y - yHat
        delta = []
        gradientW = []
        gradientB = []
        #Compute the delta values, and corresponding w/b gradients
        for i in range(self.numLayers, 1):
            prev = diff if (i == self.numLayers) else delta[0]
            deltaCurrent = np.multiply(prev, self.sigmoidPrime(self.getZ(i)))
            delta.insert(0, deltaCurrent)

            actPrev = self.getActivations(i - 1)
            wCurrent = np.dot(deltaCurrent, actPrev.transpose())
            bCurrent = deltaCurrent
            gradientW.insert(0, wCurrent)
            gradientB.insert(0, bCurrent)
        print(self.computeCost(dataNum))
        return (gradientW, gradientB)

    def SGD(self, epochs, batchSize, eta):
        dataLen = len(self.xSet)
        for iter in range(epochs):
            self.shuffleList()
            batches = [range(n, n + batchSize) for n in range(0, dataLen, batchSize)]
            for m in batches:
                self.updateBatch(m, eta)
            print("Epoch Complete")



    def updateBatch(self, batch, eta):
        wGradient = [np.zeros(w.shape) for w in self.weights]
        bGradient = [np.zeros(b.shape) for b in self.bias]

        for number in batch:
            currentBGradient, currentWGradient = self.backProp(number)
            #Element wise summation
            wGradient = [a + b for a, b in zip(wGradient, currentWGradient)]
            bGradient = [a + b for a, b in zip(bGradient, currentBGradient)]
        self.weights = [w - (eta / len(batch)) * newW for w, newW in zip(self.weights, wGradient)]
        self.bias = [b - (eta / len(batch)) *newB for b, newB in zip(self.weights, bGradient)]




    def initWeights(self):
        for i in range(2, self.numLayers):
            prev = self.getLayerDimension(i - 1)
            current = self.getLayerDimension(i)
            self.weights.append(np.random.rand(current, prev))

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidPrime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def randomShuffle(self, seed, list):
        random.seed(seed)
        for i in range(0, len(list) - 2):
            j = random.randint(0, i)
            list[i], list[j] = list[j], list[i]
        return list
    def getActivations(self, i):
        return self.activations[i + self.VALUE_SHIFT]
    def getZ(self, i):
        return self.z[i + self.VALUE_SHIFT]
    def getWeights(self, i):
        return self.weights[i + self.VALUE_SHIFT]
    def getBias(self, i):
        return self.bias[i + self.VALUE_SHIFT]
    def getLayerDimension(self, layer):
        return self.totalLayerDim[layer - 1]

    def setActivations(self, i, v):
        self.activations[i + self.VALUE_SHIFT] = v.copy()
    def setZ(self, i, v):
        self.z[i + self.VALUE_SHIFT] = v.copy()
    def setWeights(self, i, v):
        self.weights[i + self.VALUE_SHIFT] = v.copy()
    def setBias(self, i, v):
        self.bias[i + self.VALUE_SHIFT] = v.copy()

    def shuffleList(self):
        np.random.shuffle(self.dataSet)
        self.xSet = []
        self.ySet = []
        for item in self.dataSet:
            self.xSet.append(item[0])
            self.ySet.append(item[1])


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

testNN = NeuralNet(training_data, [784, 30, 10])