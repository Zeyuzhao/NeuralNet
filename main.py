import numpy as np

class NeuralNet():
    def NeuralNet(self, xSet, ySet, dim):
        self.dim = dim
        self.layers = len(self.dim)
        self.weights = []
        self.initWeights()

        self.bias = [np.random.random(b) for b in dim]
        self.activations = [np.zeros(b, 1) for b in dim]
        self.xSet = [np.array(x) for x in xSet]
        self.ySet = [np.array(y) for y in ySet]

    def fowardProp(self, dataNum):
        for i in range(self.layers - 1):


    def computeCost(self):
        pass

    def backProp(self):
        pass

    def gradientDescent(self):
        pass

    def initWeights(self):
        dim = self.dim
        for i in range(1, len(dim)):
            prev = dim[i - 1]
            current = dim[i]
            self.weights.append(np.random.rand(current, prev))

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidPrime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))
