from multilayer_perceptron import MultiLayerPerceptron, DataSet
from matplotlib import pyplot as plt
import numpy as np


def generate_data(n, trainingDataFraction):

    ind = int(np.round(n*trainingDataFraction))

    theta = np.random.random_sample(n)*2*np.pi
    r = np.random.normal(4.5, 1, n)
    xVals1 = 2*r*np.sin(theta)
    yVals1 = r*np.cos(theta)+np.sin(xVals1)
    class1 = np.vstack((xVals1, yVals1)).T
    class1Training = class1[0:ind, :]
    class1Validation = class1[ind:, :]

    mean = np.array([0, 0])
    cov = np.diag([4, 1], 0)
    class2 = np.random.multivariate_normal(mean, cov, n)
    class2[:, 1] = class2[:, 1]+np.sin(class2[:, 0])
    class2Training = class2[0:ind, :]
    class2Validation = class2[ind:, :]

    trainingObservations = np.vstack((class1Training, class2Training))
    trainingTargetClasses = np.vstack((np.zeros((class1Training.shape[0], 1)), np.ones((class2Training.shape[0], 1))))
    validationObservations = np.vstack((class1Validation, class2Validation))
    validationTargetClasses = np.vstack((np.zeros((class1Validation.shape[0], 1)), np.ones((class2Validation.shape[0], 1))))
    trainingData = DataSet(trainingObservations, trainingTargetClasses)
    validationData = DataSet(validationObservations, validationTargetClasses)

    return trainingData, validationData


def main():
    plt.close('all')
    np.random.seed(0)
    trainingData, validationData = generate_data(n=2000, trainingDataFraction=.8)
    learningRate = .1
    nHiddenNeurons = [5, 2]
    neuralNet = MultiLayerPerceptron(learningRate, trainingData, validationData, nHiddenNeurons)
    neuralNet.train(nEpochs=75, batchSize=128, momentum=.8)
    print('Accuracy is {}'.format(np.mean(neuralNet.classify(validationData.observations) == validationData.targetClasses)))
    # plot result
    xv, yv = np.meshgrid(np.arange(-7, 7, 0.02), np.arange(-5, 5, 0.02))
    observations = np.vstack((np.ravel(xv), np.ravel(yv))).T
    a = neuralNet._feed_forward(observations)
    a = a[-1][:, 1, 0]
    zeros = observations[abs(a-.5) < .01, :]
    plt.figure(1)
    plt.scatter(np.arange(len(neuralNet.validationCostHistory)), neuralNet.validationCostHistory, s=5)
    plt.title('Validation Cost vs. Epoch')
    plt.figure(2)
    mask = trainingData.targetClasses.astype(bool)[:, 0]
    plots = (plt.scatter(trainingData.observations[mask, 0], trainingData.observations[mask, 1], s=5),
             plt.scatter(trainingData.observations[~mask, 0], trainingData.observations[~mask, 1], s=5),
             plt.scatter(zeros[:, 0], zeros[:, 1], s=1))
    plt.legend(plots, ('Class 1', 'Class 2', 'Decision Boundary'))
    plt.title('Decision Boundary')


main()
