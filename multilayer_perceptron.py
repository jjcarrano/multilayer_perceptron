import numpy as np
import copy


def sigmoid(x):
    return 1/(1+np.exp(-x))


def ReLU(x):
    return np.maximum(x, 0)


def ReLUDerivative(y):
    return np.heaviside(y, 0)


class DataSet:
    # consider emulating matlab structure arrays
    def __init__(self, observations, targetClass):
        self._observations = observations
        self._targetClasses = targetClass
        self._observations.setflags(write=False)  # observations and targetClasses must be numpy arrays
        self._targetClasses.setflags(write=False)

    @property
    def observations(self):
        return self._observations

    @property
    def targetClasses(self):
        return self._targetClasses

    def shuffle(self):
        randomInd = np.random.permutation(self._observations.shape[0])
        self._observations = self._observations[randomInd, :]
        self._targetClasses = self._targetClasses[randomInd, :]


class MultiLayerPerceptron:
    def __init__(self, learningRate, trainingData, validationData, nHiddenNeurons):
        self.trainingData = copy.deepcopy(trainingData)
        self.validationData = copy.deepcopy(validationData)
        self.learningRate = learningRate
        self.nNeurons = [trainingData.observations.shape[1]]+nHiddenNeurons+[1]
        self.nLayers = len(self.nNeurons)
        self.W = self._initialize_weights()
        self.weightHistory = None
        self.validationCostHistory = None

    def _initialize_weights(self):
        np.random.seed()
        W = np.empty(self.nLayers, dtype=object)
        W[0] = np.array([])
        for iLayer in range(1, self.nLayers):
            weightMatrix = np.random.normal(0, np.sqrt(2/self.nNeurons[iLayer-1]),  # He weight initialization
                                            (self.nNeurons[iLayer], self.nNeurons[iLayer-1]))
            weightMatrix = np.hstack((np.zeros((self.nNeurons[iLayer], 1)), weightMatrix))
            W[iLayer] = weightMatrix
        return W

    def _feed_forward(self, observations):
        observations = observations[:, :, np.newaxis]
        onesArr = np.ones((observations.shape[0], 1, 1))
        activations = [np.concatenate((onesArr, observations), axis=1)]
        activationFuncs = [[]]+[ReLU]*(self.nLayers-2)+[sigmoid]
        for iLayer in range(self.nLayers-1):
            aTemp = activationFuncs[iLayer+1](self.W[iLayer+1] @ activations[iLayer])
            activations.append(np.concatenate((onesArr, aTemp), axis=1))
        return activations

    def _back_propagate(self, activations, targetClasses):
        # uses cross-entropy loss function
        costGradient = np.empty(self.nLayers, dtype=object)
        costGradient[0] = np.array([])
        delta = -self.learningRate*(activations[-1][:, 1:]-targetClasses[:, :, np.newaxis])
        costGradient[-1] = np.mean(delta @ np.transpose(activations[-2], (0, 2, 1)), axis=0)
        activationFuncDerivatives = [[]]+[ReLUDerivative]*(self.nLayers-2)
        for iLayer in range(self.nLayers-2, 0, -1):
            delta = self.W[iLayer+1].T[1:] @ delta*activationFuncDerivatives[iLayer](activations[iLayer][:, 1:])
            costGradient[iLayer] = np.mean(delta @ np.transpose(activations[iLayer-1], (0, 2, 1)), axis=0)
        return costGradient

    def train(self, nEpochs, batchSize, momentum):
        nObservations = self.trainingData.observations.shape[0]
        indices = np.append(np.arange(0, nObservations, batchSize), nObservations)
        iteration = 0
        self.validationCostHistory = np.zeros(nEpochs)
        self.weightHistory = [None]*nEpochs
        for iEpoch in range(nEpochs):
            self.trainingData.shuffle()
            for i in range(len(indices)-1):
                iteration += 1
                start = indices[i]
                end = indices[i+1]
                activations = self._feed_forward(self.trainingData.observations[start:end, :])
                newCostGradient = self._back_propagate(activations, self.trainingData.targetClasses[start:end, :])
                if iteration == 1:
                    costGradient = newCostGradient 
                else:
                    costGradient = newCostGradient*(1-momentum)+costGradient*momentum
                self.W += costGradient
            q = self._feed_forward(self.validationData.observations)
            q = q[-1][:, 1:, 0]
            self.validationCostHistory[iEpoch] = np.mean(-self.validationData.targetClasses*np.log(q)
                                                         -(1-self.validationData.targetClasses)*np.log(1-q))
            self.weightHistory[iEpoch] = self.W

    def classify(self, observations):
        activations = self._feed_forward(observations)
        predictedClasses = np.round(activations[-1][:, 1:, 0])
        return predictedClasses
