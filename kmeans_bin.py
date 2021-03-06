import h5py
import os
import random
import numpy as np
#import scipy
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from numpy.linalg import norm
import model


def randomSampleFeatures(song, numSamples):
    chosen = []
    remaining = len(song.featureVectors)
    for i in range(len(song.featureVectors)):
        prob = numSamples/float(remaining)
        if random.random()<prob:
            chosen.append(song.featureVectors[i])
            numSamples-=1
        remaining-=1
    return chosen

'''todo: make it so you return numVectors exactly.
As it is, it returns a number <= numVectors, basically
that number rounded down to be a multiple of numSongs.'''

def getFeatureVectors(songs, numVectors):
    perSong = numVectors/(len(songs))
    featureVectors = []
    for song in songs:
        featureVectors+=randomSampleFeatures(song, perSong)
    return featureVectors

def getCentroids(trainingSongs, num_centroids):
    
    total_feature_vectors = 3000

    songfeatures = getFeatureVectors(trainingSongs, total_feature_vectors)
    songfeatures_ndarray = np.array(songfeatures)
    '''Whitening divides each feature by its std-dev to give unit variance.
    This is a necessary step before doing the scipy k-means.
    Note that to classify new  training examples, I'll need to
    normalize those input features with the same ratios which I'll need to
    calculate.
    '''
    song_whitened = whiten(songfeatures_ndarray)
    ratios =  songfeatures_ndarray[0] / song_whitened[0]
    song_codebook, song_distortion = kmeans(song_whitened, num_centroids)
    centroids = song_codebook * ratios
    return centroids, song_distortion



def getFeatureError(vector, centroids):
    return min(norm(vector- centroid, 2) for centroid in centroids)

def getSongError(song, centroids):
    total = 0
    for vector in song.featureVectors:
        error = getFeatureError(vector, centroids)
        total+=error
    return total
        
def classify(song, firstCentroids, secondCentroids):
    e1 = getSongError(song, firstCentroids)
    e2 = getSongError(song, secondCentroids)
    if e1<e2:
        return 'first'
    else:
        return 'second'

def getValidationError(firstTest, secondTest, firstCentroids, secondCentroids):
    firstRight = 0
    firstWrong = 0
    secondRight = 0
    secondWrong = 0
    for song in firstTest:
        if classify(song, firstCentroids, secondCentroids)=='first':
            firstRight+=1
        else:
            firstWrong+=1
    for song in secondTest:
        if classify(song, firstCentroids, secondCentroids)=='second':
            secondRight+=1
        else:
            secondWrong+=1
    return firstRight, firstWrong, secondRight, secondWrong

def findModel(numCentroids):

    print "/n/nNumber of centroids: ", numCentroids
    rock_centroids, rock_distortion = getCentroids(trainRock, numCentroids)
    jazz_centroids, jazz_distortion = getCentroids(trainJazz, numCentroids)


    print "Validating with centroids = ", numCentroids
    jazzRight, jazzWrong, rockRight, rockWrong = getValidationError(testJazz, testRock, jazz_centroids, rock_centroids)
    print 'rock'
    print "rockRight: ", rockRight
    print "rockWrong: ", rockWrong
    print
    print 'jazz'
    print "correct: ", jazzRight
    print "incorrect: ", jazzWrong
    
rocksongs, jazzsongs = model.getData(250)
trainRock = rocksongs[:200]
testRock = rocksongs[200:]
trainJazz = jazzsongs[:200]
testJazz = jazzsongs[200:]

findModel(5)
findModel(10)
findModel(20)
findModel(40)
