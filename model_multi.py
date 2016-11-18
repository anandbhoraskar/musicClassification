import os
import cPickle as pickle
import numpy as np
from song_multi import Song


def fillSongs(numNeeded):
    rocksongs = []
    jazzsongs = []
    hiphopsongs = []
    popsongs = []
    metalsongs = []

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
                    if len(rocksongs) >= numNeeded and len(jazzsongs) >= numNeeded and len(popsongs) >= numNeeded and len(hiphopsongs) >= numNeeded and len(metalsongs) >= numNeeded:
                        return rocksongs[:numNeeded], hiphopsongs[:numNeeded], popsongs[:numNeeded], jazzsongs[:numNeeded], metalsongs[:numNeeded]
                    needJazz = len(jazzsongs) < numNeeded
                    needRock = len(rocksongs) < numNeeded
                    needPop = len(popsongs) < numNeeded
                    needHipHop = len(hiphopsongs) < numNeeded
                    needMetal = len(metalsongs) < numNeeded
                    currsong = Song(fpath, needRock, needHipHop, needPop, needJazz, needMetal)
                    if currsong.genre == 'metal' and needMetal:
                        if (len(currsong.featureVectors) > 0):
                            metalsongs.append(currsong)
                    elif currsong.genre == 'jazz' and needJazz:
                        if (len(currsong.featureVectors) > 0):
                            jazzsongs.append(currsong)
                    elif currsong.genre == 'pop' and needPop:
                        if (len(currsong.featureVectors) > 0):
                            popsongs.append(currsong)
                    elif currsong.genre == 'hip hop' and needHipHop:
                        if (len(currsong.featureVectors) > 0):
                            hiphopsongs.append(currsong)
                    elif currsong.genre == 'rock' and needRock:
                        if (len(currsong.featureVectors) > 0):
                            rocksongs.append(currsong)

    return rocksongs, hiphopsongs, popsongs, jazzsongs, metalsongs
    #return tmp

    #print tmp
    #print "paused"
    #raw_input()


def getData(numPerGenre):
    if os.path.exists("jazz.pkl") and os.path.exists("rock.pkl") and os.path.exists("hiphop.pkl") and os.path.exists("pop.pkl") and os.path.exists("metal.pkl"):
        with open('jazz.pkl', 'rb') as inp:
            jazzsongs = pickle.load(inp)
            print 'Loaded jazz songs'
        with open('hiphop.pkl', 'rb') as inp:
            hiphopsongs = pickle.load(inp)
            print 'Loaded hip hop songs'
        with open('pop.pkl', 'rb') as inp:
            popsongs = pickle.load(inp)
            print 'Loaded pop songs'
        with open('metal.pkl', 'rb') as inp:
            metalsongs = pickle.load(inp)
            print 'Loaded metal songs'
        with open('rock.pkl', 'rb') as inp:
            rocksongs = pickle.load(inp)
            print 'Loaded rock songs'
    else:
        rocksongs, hiphopsongs, popsongs, jazzsongs, metalsongs = fillSongs(numPerGenre)

        output = open('jazz.pkl', 'wb')
        pickle.dump(jazzsongs, output)
        output.close()
        output = open('rock.pkl', 'wb')
        pickle.dump(rocksongs, output)
        output.close()
        output = open('pop.pkl', 'wb')
        pickle.dump(popsongs, output)
        output.close()
        output = open('hiphop.pkl', 'wb')
        pickle.dump(hiphopsongs, output)
        output.close()
        output = open('metal.pkl', 'wb')
        pickle.dump(metalsongs, output)
        output.close()

    return rocksongs[:numPerGenre], hiphopsongs[:numPerGenre], popsongs[:numPerGenre], jazzsongs[:numPerGenre], metalsongs[:numPerGenre]
    assert (len(rocksongs) == numPerGenre and len(jazzsongs) == numPerGenre and len(hiphopsongs) == numPerGenre and len(popsongs) == numPerGenre and len(metalsongs) == numPerGenre)  # if false then need more data


def songToInput(song):
    return song.meanVector + \
           [item for sublist in song.covarianceMatrix for item in sublist]


def songToResultVector(song):
    if song.genre == 'rock':
        return 0
    elif song.genre == 'hip hop':
        return 1
    elif song.genre == 'pop':
        return 2
    elif song.genre == 'jazz':
        return 3
    elif song.genre == 'metal':
        return 4
    else:
        return 5

'''
def main():
    rocksongs, hiphopsongs, popsongs, jazzsongs, metalsongs = getData(1000)
    
    with open('jazz1.pkl', 'rb') as inp:
        jazzsongs = pickle.load(inp)
    with open('metal1.pkl', 'rb') as inp:
        metalsongs = pickle.load(inp)

    training_input = np.asarray([songToInput(song) for song in rocksongs[:200] + hiphopsongs[:200] + popsongs[:200] + jazzsongs[:200] + metalsongs[:200]])
    training_result = np.asarray([songToResultVector(song) for song in rocksongs[:200] + hiphopsongs[:200] + popsongs[:200] + jazzsongs[:200] + metalsongs[:200]])

    test_input = np.asarray([songToInput(song) for song in rocksongs[200:] + hiphopsongs[200:] + popsongs[200:] + jazzsongs[200:] + metalsongs[200:]])
    test_result = np.asarray([songToResultVector(song) for song in rocksongs[200:] + hiphopsongs[200:] + popsongs[200:] + jazzsongs[200:] + metalsongs[200:]])

    # if __name__ == "__main__":
    #     main()
'''