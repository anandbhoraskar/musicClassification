import random
import numpy as np
import model_multi

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


def getTrainingData(songs, perSong, div):
    train = []
    test = []
    for i in range(len(songs)):
        features = getFeatureVectors(songs[i], perSong)
        train.append(features[:div])
        test.append(features[div:])

    def shuffle(inpdata):
        X = np.empty((0, perSong * 12))
        Y = []
        for i in range(len(inpdata)):
            X = np.append(X, inpdata[i], axis=0)
            Y += [i] * len(inpdata[i])
        data = zip(X, Y)
        random.shuffle(data)
        X, Y = zip(*data)
        return [X, Y]

    training_data = shuffle(train)
    testing_data = shuffle(test)
    return training_data, testing_data

def confusion_matrix(actual, pred):
    matrix = np.zeros((5, 5))
    for (x, y) in zip(actual, pred):
        matrix[x, y] += 1
    return matrix  # row : actual col: predicted

# rockLabel = 0
# hiphopLabel = 1
# popLabel = 2
# jazzLabel = 3
# metalLabel = 4

songs = list(model_multi.getData(250))
training_data, testing_data = getTrainingData(songs, 100, 200)
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(600, 300, 150, 30))

clf.fit(training_data[0], training_data[1])
pred = clf.predict(testing_data[0])
print sum(testing_data[1] == pred), (float)(sum(testing_data[1] == pred))/len(pred)
print confusion_matrix(testing_data[1], pred)