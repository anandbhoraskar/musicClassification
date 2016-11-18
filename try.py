import os
import h5py
import numpy as np
from sklearn import svm
from scipy.cluster.vq import whiten
#from copy import deepcopy

import model_multi as model
from Song import Song

#rocksongs, hiphopsongs, popsongs, jazzsongs, metalsongs

def getSVM_OVR(rockTrain, hiphopTrain, popTrain, jazzTrain, metalTrain):
	trainMatrix = []
	labels = []

	rockLabel = 0
	hiphopLabel = 1
	popLabel = 2
	jazzLabel = 3
	metalLabel = 4

	for featureVector in rockTrain:
		trainMatrix.append(featureVector)	
		labels.append(rockLabel)

	# print type(trainMatrix)
	# print type(labels)

	for featureVector in hiphopTrain:
		trainMatrix.append(featureVector)
		labels.append(hiphopLabel)
	
	for featureVector in popTrain:
		trainMatrix.append(featureVector)
		labels.append(popLabel)	
	
	for featureVector in jazzTrain:
		trainMatrix.append(featureVector)
		labels.append(jazzLabel)

	for featureVector in metalTrain:
		trainMatrix.append(featureVector)
		labels.append(metalLabel)

	svmOVR = svm.LinearSVC()
	print "Training SVM (one-vs-rest) on ", len(trainMatrix), "examples..."

	svmOVR.fit(trainMatrix,labels)
	return svmOVR

# def classify(song, SVMobj)


def predictSVM(svmOVR, rockTest, hiphopTest, popTest, jazzTest, metalTest):
	testMatrix = []

	for featureVector in rockTest:
		testMatrix.append(featureVector)
		
	for featureVector in hiphopTest:
		testMatrix.append(featureVector)
		
	for featureVector in popTest:
		testMatrix.append(featureVector)
		
	for featureVector in jazzTest:
		testMatrix.append(featureVector)
		
	for featureVector in metalTest:
		testMatrix.append(featureVector)
		
	print "Predicting values..."
	testPredict = svmOVR.predict(testMatrix)

	return testPredict

rocksongs, hiphopsongs, popsongs, jazzsongs, metalsongs = model.getData(2)
trainRock = rocksongs[:1]
testRock = rocksongs[1:]
trainHipHop = hiphopsongs[:1]
testHipHop = hiphopsongs[1:]
trainPop = popsongs[:1]
testPop = popsongs[1:]
trainJazz = jazzsongs[:1]
testJazz = jazzsongs[1:]
trainMetal = metalsongs[:1]
testMetal = metalsongs[1:]

# testResult = []
# for i in range(1,5):
# 	testResult.append(0)
# for i in range(6,10):
# 	testResult.append(1)
# for i in range(11,15):
# 	testResult.append(2)
# for i in range(16,20):
# 	testResult.append(3)
# for i in range(20,25):
# 	testResult.append(4)		

print type(trainRock)
# print type(labels)

songSVM_OVR = getSVM_OVR(trainRock, trainHipHop, trainPop, trainJazz, trainMetal)
testPredict = predictSVM(songSVM_OVR, testRock, testHipHop, testPop, testJazz, testMetal)

correct = np.sum(testPredict == testResult)

print "Number of correct values out of 250 is ", correct
