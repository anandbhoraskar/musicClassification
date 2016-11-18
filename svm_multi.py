import os
import h5py
import numpy as np
from sklearn import svm
from scipy.cluster.vq import whiten
from copy import deepcopy

import model_multi as model
from Song import Song

#rocksongs, hiphopsongs, popsongs, jazzsongs, metalsongs

def getTrainedSVM(whiteSongTrain):
    trainMatrix = []
    labels = []

    for i in range(5):
        for featureVector in whiteSongTrain[i]:
            trainMatrix.append(featureVector)
            labels.append(i)

    svmOVR = svm.LinearSVC()
    print "Training SVM (one-vs-rest) on ", len(trainMatrix), "feature vectors..."

    svmOVR.fit(trainMatrix,labels)
    return svmOVR


def classify(song, classifier):
    total = np.zeros(5)
    # record= []
    for featureVector in song.featureVectors:
        decision = classifier.decision_function([featureVector])
        prediction = np.argmax(decision[0])  # the confidence value
        total[prediction] += 1
    # record.append(prediction)
    # negative means 0 means rock, positive means 1 means jazz
    # print "total: ", total, "\nrecord: ", record
    netPrediction = np.argmax(total)
    return netPrediction


# def getClassifierAccuracy(rockTest, jazzTest, classifier):
#   rockRight = 0
#   rockWrong = 0
#   jazzRight = 0
#   jazzWrong = 0
#
#   for song in rockTest:
#       if classify(song, classifier) == ROCK_LABEL:
#           rockRight += 1
#       else:
#           rockWrong += 1
#
#   for song in jazzTest:
#       if classify(song, classifier) == JAZZ_LABEL:
#           jazzRight += 1
#       else:
#           jazzWrong += 1
#
#   return rockRight, rockWrong, jazzRight, jazzWrong


def getFeatures(songList):
    features = []
    for song in songList:
        for vector in song.featureVectors:
            features.append(vector)
    return features


def updateSongs(whitenedFeatures, originalSongs):
    whitenedSongs = deepcopy(originalSongs)
    curr = 0
    for songIndex in range(len(originalSongs)):
        numFeatures = len(originalSongs[songIndex].featureVectors)
        newFeatures = whitenedFeatures[curr:curr + numFeatures]
        whitenedSongs[songIndex].featureVectors = newFeatures
        curr = curr + numFeatures
    return whitenedSongs

def getClassifierAccuracy(whiteTest, classifier):
    label = ['rock', 'hip hop', 'pop', 'jazz', 'metal']
    for i in range(5):
        correct = 0
        incorrect = 0
        for song in whiteTest[i]:
            if classify(song, classifier) == i:
                correct+=1
            else:
                incorrect+=1
        print label[i] + ': '
        print 'Correct = ' , correct
        print 'Incorrect = ' , incorrect
        print

def getClassification(whiteTest, classifier):
    actual = []
    pred = []
    for i in range(5):
        for song in whiteTest[i]:
            actual += [i]
            pred += [classify(song, classifier)]
    return actual, pred

def confusion_matrix(actual, pred):
    matrix = np.zeros((5, 5))
    for (x, y) in zip(actual, pred):
        matrix[x, y] += 1
    return matrix  # row : actual col: predicted
# ROCK_LABEL = 0
# JAZZ_LABEL = 1
#
# numTotal = 100
# numTrain = 80
#
# rocksongs, jazzsongs = getData(numTotal)

##allsongs_whitened = whiten(np.array(rocksongs+jazzsongs))
##whiterocksongs = allsongs_whitened[:len(rocksongs)]
##whitejazzsongs = allsongs_whitened[len(rocksongs):]

def get_white_data(songs, numTrain):
    songTrain = songs[:numTrain]
    songTest = songs[numTrain:]
    songTrainFeatures = getFeatures(songTrain)
    songTestFeatures = getFeatures(songTest)
    whitened_features = whiten(np.array(songTrainFeatures + songTestFeatures))
    whiteSongTrain = whitened_features[:len(songTrainFeatures)]
    whiteSongTest = whitened_features[len(songTrainFeatures):len(songTrainFeatures) + len(songTestFeatures)]
    whiteTest = updateSongs(whiteSongTest, songTest)
    assert (len(songTrainFeatures) == len(whiteSongTrain))
    assert (len(songTestFeatures) == len(whiteSongTest))
    return whiteSongTrain, whiteSongTest, whiteTest

rocksongs, hiphopsongs, popsongs, jazzsongs, metalsongs = model.getData(25)
numTrain = 20
train = []
whiteSongTrain = []
whiteSongTest = []
whiteTest = []

train += [rocksongs]
train += [hiphopsongs]
train += [popsongs]
train += [jazzsongs]
train += [metalsongs]

for i in range(5):
    a,b,c = get_white_data(train[i], numTrain)
    whiteSongTrain += [a]
    whiteSongTest += [b]
    whiteTest += [c]


#
# rockTrain = rocksongs[:numTrain]
# jazzTrain = jazzsongs[:numTrain]
# rockTest = rocksongs[numTrain:]
# jazzTest = jazzsongs[numTrain:]
#
# rockTrainFeatures = getFeatures(rockTrain)
# rockTestFeatures = getFeatures(rockTest)
# jazzTrainFeatures = getFeatures(jazzTrain)
# jazzTestFeatures = getFeatures(jazzTest)
#
# whitened_features = whiten(np.array(rockTrainFeatures + rockTestFeatures + jazzTrainFeatures + jazzTestFeatures))
#
# endRockTrain = len(rockTrainFeatures)
# endRockTest = endRockTrain + len(rockTestFeatures)
# endJazzTrain = endRockTest + len(jazzTrainFeatures)
# endJazzTest = endJazzTrain + len(jazzTestFeatures)
#
# whiteRockTrain = whitened_features[:len(rockTrainFeatures)]
# whiteRockTest = whitened_features[endRockTrain:endRockTest]
# whiteJazzTrain = whitened_features[endRockTest:endJazzTrain]
# whiteJazzTest = whitened_features[endJazzTrain:endJazzTest]
#
# whiteRockTestSongs = updateSongs(whiteRockTest, rockTest)
# whiteJazzTestSongs = updateSongs(whiteJazzTest, jazzTest)
#
# assert (len(rockTrainFeatures) == len(whiteRockTrain))
# assert (len(rockTestFeatures) == len(whiteRockTest))
# assert (len(jazzTrainFeatures) == len(whiteJazzTrain))
# assert (len(jazzTestFeatures) == len(whiteJazzTest))

print "Getting Classifier"
classifier = getTrainedSVM(whiteSongTrain)

print "Validating accuracy"
actual, pred = getClassification(whiteTest, classifier)
print confusion_matrix(actual, pred)
# print "rockRight: ", rockRight
# print "rockWrong: ", rockWrong
# print "jazzRight: ", jazzRight
# print "jazzWrong: ", jazzWrong

'''
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
'''