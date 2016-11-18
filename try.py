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

def getFeatures(songList):
    features=[]
    for song in songList:
        for item in song.featureVectors:
            features.append(item)
    return features


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

rocksongs, hiphopsongs, popsongs, jazzsongs, metalsongs = model.getData(25)
trainRock = getFeatures(rocksongs[:20])
testRock = getFeatures(rocksongs[20:])
trainHipHop = getFeatures(hiphopsongs[:20])
testHipHop = getFeatures(hiphopsongs[20:])
trainPop = getFeatures(popsongs[:20])
testPop = getFeatures(popsongs[20:])
trainJazz = getFeatures(jazzsongs[:20])
testJazz = getFeatures(jazzsongs[20:])
trainMetal = getFeatures(metalsongs[:20])
testMetal = getFeatures(metalsongs[20:])

testResult = []
for i in range(0,5):
	testResult.append(0)
for i in range(0,5):
	testResult.append(1)
for i in range(0,5):
	testResult.append(2)
for i in range(0,5):
	testResult.append(3)
for i in range(0,5):
	testResult.append(4)		

# testResult = [0,1,2,3,4]

# print type(trainRock)
# print type(labels)

songSVM_OVR = getSVM_OVR(trainRock, trainHipHop, trainPop, trainJazz, trainMetal)
testPredict = predictSVM(songSVM_OVR, testRock, testHipHop, testPop, testJazz, testMetal)

correct = np.sum(testPredict == testResult)

print testResult
print testPredict

print "Number of correct values out of 25 is ", correct
