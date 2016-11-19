import random
import numpy as np
import model

from sklearn.neural_network import MLPClassifier


# def randomSampleFeatures(song, numSamples):
#     chosen = []
#     remaining = len(song.featureVectors)
#     for i in range(len(song.featureVectors)):
#         prob = numSamples / float(remaining)
#         if random.random() < prob:
#             chosen.append(song.featureVectors[i])
#             numSamples -= 1
#         remaining -= 1
#     return chosen

def randomSampleFeatures(song, numSamples):
    chosen = []
    for i, fv in enumerate(song.featureVectors):
        if i < numSamples:
            chosen.append(fv)
        elif i >= numSamples and random.random() < numSamples / float(i + 1):
            replace = random.randint(0, len(chosen) - 1)
            chosen[replace] = fv
    return chosen


def getFeatureVectors(songs, perSong):
    featureVectors = np.empty((0, perSong * 12))
    for song in songs:
        fv = randomSampleFeatures(song, perSong)
        if len(fv) == perSong:
            featureVectors = np.append(featureVectors, np.reshape(fv, (1, perSong * 12)), axis=0)
    return featureVectors


def getTrainingData(rocksongs, jazzsongs, perSong, div):
    rockfeatures = getFeatureVectors(rocksongs, perSong)
    jazzfeatures = getFeatureVectors(jazzsongs, perSong)
    trainRock = rockfeatures[:div]
    testRock = rockfeatures[div:]
    trainJazz = jazzfeatures[:div]
    testJazz = jazzfeatures[div:]

    def shuffle(R, J):
        X = np.empty((0, perSong * 12))
        X = np.append(X, R, axis=0)
        X = np.append(X, J, axis=0)
        Y = [0] * len(R) + [1] * len(J)
        data = zip(X, Y)
        random.shuffle(data)
        X, Y = zip(*data)
        return [X, Y]

    training_data = shuffle(trainRock, trainJazz)
    testing_data = shuffle(testRock, testJazz)
    return training_data, testing_data

def confusion_matrix(actual, pred):
    matrix = np.zeros((5, 5))
    for (x, y) in zip(actual, pred):
        matrix[x, y] += 1
    return matrix  # row : actual col: predicted

rocksongs, jazzsongs = model.getData(250)
training_data, testing_data = getTrainingData(rocksongs, jazzsongs, 90, 200)
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100 * 12 * 2, 120, 60, 30))

clf.fit(training_data[0], training_data[1])
pred = clf.predict(testing_data[0])

print sum(testing_data[1] == pred), (float)(sum(testing_data[1] == pred))/len(pred)
print confusion_matrix(testing_data[1], pred)