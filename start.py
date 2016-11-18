# CS 725: Music Genre Classification

import h5py				# To read HDF5 and H5 files
import numpy as np

'''
with h5py.File('millionsongsubset_full/MillionSongSubset/data/A/A/A/TRAAAAW128F429D538.h5','r') as hf:
	
	print('List of arrays: \n',hf.keys())

	data = hf.get('analysis')
	np_analysis = np.array(data)
	print('Shape of analysis:', np.shape(np_analysis))

	data = hf.get('metadata')
	np_metadata = np.array(data)
	print('Shape of metadata:', np.shape(np_metadata))

	data = hf.get('musicbrainz')
	np_musicbrainz = np.array(data)
	print('Shape of musicbrainz:', np.shape(np_musicbrainz))

	a = np.concatenate((np_analysis,np_metadata,np_musicbrainz), axis=0)
	np.savetxt("check.csv", a, delimiter=",")
'''

fname = 'MillionSongSubset/data/A/A/A/TRAAAAW128F429D538.h5'
f = h5py.File(fname, 'r')

tags = f['metadata']['artist_terms']
alltags = []
for tag in tags:
	alltags.append(tag)

# print alltags

mfcc2d = f['analysis']['segments_timbre']
# print type(mfcc2d)

featureVectors = []
for i in range(len(mfcc2d)):
	item = mfcc2d[i]
	featureVectors.append(item)

fV = np.asarray(featureVectors)

print fV
print np.size(fV)
