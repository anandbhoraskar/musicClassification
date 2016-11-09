import os
import cPickle as pickle
import numpy as np
from Song import Song


def fillSongs(numNeeded):
    rocksongs = []
    jazzsongs = []

    total = 0
    songdirtop = "MillionSongSubset/data"
    for subdirtop in os.listdir(songdirtop):
        songdir = songdirtop + "/" + subdirtop
        for subdir in os.listdir(songdir):
            name = songdir + "/" + subdir
            try:
                directory = os.listdir(name)
                print "Directory is ", directory
            except Exception, e:
                print "Continuing because " + name + " is not a directory"
                continue
            for ssubdir in directory[:20]:
                name2 = name + "/" + ssubdir
                print name2
                try:
                    print "Trying to get stuff"
                    directory2 = os.listdir(name2)
                    print "Internal directory is ", directory2
                    print len(directory2)
                except Exception, e:
                    "Continuing because " + name2 + " is not a directory in 2"
                    continue
                for fname in directory2:
                    fpath = name2 + "/" + fname
                    print fpath
                    print min(len(rocksongs), len(jazzsongs))
                    if len(rocksongs) >= numNeeded and len(jazzsongs) >= numNeeded:
                        return rocksongs[:numNeeded], jazzsongs[:numNeeded]
                    needJazz = len(jazzsongs) < numNeeded
                    needRock = len(rocksongs) < numNeeded
                    currsong = Song(fpath, needJazz, needRock)
                    if currsong.genre == 'jazz' and needJazz:
                        if (len(currsong.featureVectors) > 0):
                            jazzsongs.append(currsong)
                    elif currsong.genre == 'rock' and needRock:
                        if (len(currsong.featureVectors) > 0):
                            rocksongs.append(currsong)

    return rocksongs, jazzsongs


def getData(numPerGenre):
    if os.path.exists("jazz.pkl") and os.path.exists("rock.pkl"):
        with open('jazz.pkl', 'rb') as inp:
            jazzsongs = pickle.load(inp)
            print 'Loaded jazz songs'
        with open('rock.pkl', 'rb') as inp:
            rocksongs = pickle.load(inp)
            print 'Loaded rock songs'
    else:
        rocksongs, jazzsongs = fillSongs(numPerGenre)
    return rocksongs[:numPerGenre], jazzsongs[:numPerGenre]
    assert (len(rocksongs) == numPerGenre and len(jazzsongs) == numPerGenre)  # if false then need more data


def songToInput(song):
    return song.meanVector + \
           [item for sublist in song.covarianceMatrix for item in sublist]


def songToResultVector(song):
    if song.genre == 'jazz':
        return 0
    elif song.genre == 'rock':
        return 1
    else:
        return 2

    # def main():
    metalsongs, jazzsongs = getData(1000)
    output = open('jazz.pkl', 'wb')
    pickle.dump(jazzsongs, output)
    output.close()
    output = open('metal.pkl', 'wb')
    pickle.dump(metalsongs, output)
    output.close()

    with open('jazz1.pkl', 'rb') as inp:
        jazzsongs = pickle.load(inp)
    with open('metal1.pkl', 'rb') as inp:
        metalsongs = pickle.load(inp)

    training_input = np.asarray([songToInput(song) for song in metalsongs[:200] + jazzsongs[:200]])
    training_result = np.asarray([songToResultVector(song) for song in metalsongs[:200] + jazzsongs[:200]])

    test_input = np.asarray([songToInput(song) for song in metalsongs[200:] + jazzsongs[200:]])
    test_result = np.asarray([songToResultVector(song) for song in metalsongs[200:] + jazzsongs[200:]])

    # if __name__ == "__main__":
    #     main()
