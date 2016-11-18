import h5py

class Song:
    def __init__(self, fname, needJazz, needRock):
        try:
            f = h5py.File(fname, 'r')
        except Exception, e:
            self.featureVectors = []
            self.genre = 'neither'
            return
        self.fname = fname
        self.tags = self.getTags(f)
        self.genre = self.getGenre(self.tags)
        if self.genre == 'neither':
            f.close()
            return
        if self.genre == 'jazz' and not needJazz:
            f.close()
            return
        if self.genre == 'rock' and not needRock:
            f.close()
            return
        self.featureVectors, self.confidences = self.getFeatureVectorsAndConf(f)
        self.startTimes = self.getStartTimes(f)
        self.segments_pitches = self.getSegmentsPitch(f)
        self.segments_start = self.getSegmentsStart(f)
        self.segments_loudness_max = self.getSegmentsLoudnessMax(f)
        self.segments_loudness_start = self.getSegmentsLoudnessStart(f)
        if (len(self.featureVectors) == 0):
            f.close()
            return
        self.meanVector = self.getMeanVector()
        self.covarianceMatrix = self.getCovarianceMatrix()

        f.close()
        # self.confidences = self.getConfidences(f)

        # self.avgMFCC, self.maxMFCC, self.minMFCC = self.getAvgMFCC(f)
        # print "done constructing"
        # read in the song here.
        # Attributes: genre, fname, avgs for each mfcc

    def getFeatureVectorsAndConf(self, f):
        mfcc2d = f['analysis']['segments_timbre']
        confidences = f['analysis']['segments_confidence']

        confValues = []
        featureVectors = []

        for i in range(len(mfcc2d)):
            conf = confidences[i]
            # if conf<.6:
            #     continue
            item = mfcc2d[i]
            confValues.append(conf)
            featureVectors.append(item)

        return featureVectors, confValues

    def getSegmentsPitch(self, f):
        segments_pitch = []
        segments_pitches = f['analysis']['segments_pitches']
        for i in range(len(segments_pitches)):
            segments_pitch.append(segments_pitches[i])

        return segments_pitch

    def getSegmentsStart(self, f):
        segments_start = []
        segments_startz = f['analysis']['segments_start']
        for i in range(len(segments_startz)):
            segments_start.append(segments_startz[i])

        return segments_start

    def getSegmentsLoudnessStart(self, f):
        segments_start = []
        segments_startz = f['analysis']['segments_loudness_start']
        for i in range(len(segments_startz)):
            segments_start.append(segments_startz[i])

        return segments_start

    def getSegmentsLoudnessMax(self, f):
        segments_start = []
        segments_startz = f['analysis']['segments_loudness_max']
        for i in range(len(segments_startz)):
            segments_start.append(segments_startz[i])

        return segments_start

    ##    def getConfidences(self, f):
    ##
    ##        confidences = f['analysis']['segments_confidence']
    ##        confValues = []
    ##        for confidence in confidences:
    ##            confValues.append(confidence)
    ##        return confValues

    def getTags(self, f):
        tags = f['metadata']['artist_terms']
        alltags = []
        for tag in tags:
            alltags.append(tag)
        return alltags

    def getStartTimes(self, f):
        startTimes = []
        confidences = f['analysis']['segments_confidence']
        start = f['analysis']['segments_start']
        for i in range(len(start)):
            conf = confidences[i]
            # if conf<.6:
            #     continue
            startTimes.append(start[i])
        return startTimes

    def getGenre(self, tags):
        top5 = tags[:5]
        numjazz = 0
        numrock = 0
        numhip = 0
        numpop = 0
        nummetal = 0
        for tag in top5:
            if 'rock' in tag:
                numrock += 1
            if 'hip hop' in tag:
                numhip += 1
            if 'pop' in tag:
                numpop += 1
            if 'jazz' in tag:
                numjazz += 1
            if 'metal' in tag:
                nummetal += 1
        if numjazz >= 1 and nummetal == 0 and numrock == 0 and numhip == 0 and numpop == 0:
            return 'jazz'
        if numrock >= 1:
            return 'rock'
        return 'neither'

    # def getGenre(self, tags):
    #     print self.tags
    #     genreName = raw_input("Enter category name : ");
    #     if(genreName == "j" or genreName == "ja" or  genreName == "jaz" or genreName == "jazz"):
    #         print genreName
    #         return 'jazz'

    #     if(genreName == "r" or genreName == "ro" or genreName == "roc" or genreName == "rock"):
    #         print genreName
    #         return 'rock'

    #     print genreName
    #     return 'neither'

    def getAvgMFCC(self, f):
        mfcc2d = f['analysis']['segments_timbre']
        confidences = f['analysis']['segments_confidence']
        numsegments = mfcc2d.shape[0]
        numcoeff = mfcc2d.shape[1]
        totalcoeff = [0] * numcoeff
        maxcoeff = [0] * numcoeff
        mincoeff = [0] * numcoeff

        total = 0
        for i in range(len(mfcc2d)):
            conf = confidences[i]
            if conf < .6:
                continue
            total += 1
            segment = mfcc2d[i]
            for i in range(numcoeff):
                totalcoeff[i] += segment[i]
                maxcoeff[i] = max(maxcoeff[i], segment[i])
                mincoeff[i] = min(mincoeff[i], segment[i])

        avgcoeff = [item / float(total) for item in totalcoeff]

        for i in range(numcoeff):
            avgcoeff[i] = round(avgcoeff[i], 1)
            mincoeff[i] = round(mincoeff[i], 1)
            maxcoeff[i] = round(maxcoeff[i], 1)

        return avgcoeff, maxcoeff, mincoeff

    def getMeanVector(self):
        meanVector = []
        for i in range(len(self.featureVectors[0])):
            numerator = sum(item[i] for item in self.featureVectors)
            numerator = float(numerator / len(self.featureVectors))
            meanVector.append(numerator)
        return meanVector

    def getExpectedValue(self, vector):
        expVal = sum(vector)
        expVal = float(expVal / len(vector))
        return expVal

    def addtoMeanVector(self, vector):
        for i in range(len(vector)):
            self.meanVector[i] += vector[i]

    def getCovarianceMatrix(self):
        covarianceMatrix = []
        for i in range(len(self.featureVectors[0])):
            covarianceMatrix.append([]);
            currentVector = []
            for item in self.featureVectors:
                currentVector.append(item[i] - self.meanVector[i])
            for j in range(len(self.featureVectors[0])):
                innerCurrent = currentVector[:]
                for k in range(len(self.featureVectors)):
                    innerCurrent[k] = currentVector[k] * (self.featureVectors[k][j] - self.meanVector[j]);
                covarianceMatrix[i].append(self.getExpectedValue(innerCurrent));
        return covarianceMatrix

