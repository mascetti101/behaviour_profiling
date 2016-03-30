import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from array import *
from clustering.SignalClustering import SignalsClustering


__author__ = 'Paolo'

""" Class used to create words from continous signals """

class WordData():
	def __init__(self, data):
		self.dataset = data
		self.models = {}
		## max 10 cluster as assumption
		self.dictionary = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
		self.n_bins = 5
		self.discretization_model = []

	def cluster_kmeans_signal(self, signal, nClusters, featureName):
		"""
		Not used
		:param signal: matrix representing a signal, needs to be applied reshape(-1, 1)
		:param nClusters: number of clusters
		:return: predicted labels
		"""
		#(TODO): save model for clustering for each feature
		model = KMeans(n_clusters= nClusters)
		model.fit_predict(signal)
		self.models[featureName] = model
		return model.labels_

	def label_to_char(self, intLabel):
		"""
		:param intLabel: labels predicted by cluster algorithm
		:return: for each cluster int identifier returns the corresponding char
		"""
		signalsChar = array('c')
		for l in intLabel:
			char = self.dictionary[l]
			signalsChar.append(char)
		labelsChar = signalsChar.tolist()
		return labelsChar

	def create_words(self,dataset):
		"""
		:param dataset: Dataframe containing columns that we need to convert into textual values
		:return: list of words representing the original dataset
		"""
		signDiscr = SignalsClustering()

		for column in dataset.columns:
			# print "column", column

			#series = dataset[column].as_matrix()
			series = dataset[column]

			#(TODO): Remove prints
			# print "valori series:" , dataset[column]
			# print "valori matrice:", series

			""" Call to method used for discretization of signal """
			#labels = self.cluster_kmeans_signal(series.reshape(-1, 1),5, column)

			thresholds = signDiscr.signal_thresholds(series, self.n_bins)
			labels = signDiscr.assign(series,thresholds)

			""" Save discretization setting for each signal processed """
			model = {
			  'feature_name': column,
			  'thresholds' : thresholds
			}
			self.discretization_model.append(model)

			# print labels

			if dataset.columns.get_loc(column) == 0:
				newl = self.label_to_char(labels)
			else:
				charLabels = self.label_to_char(labels)
				""" char concatenation to build words for each instant """
				newl =[m+n for m,n in zip(newl,charLabels)]

		# print newl

		return newl

	def create_text_corpus(self,data):
		"""
		:param data: dataset containing at least Trip_ID and Word columns
		:return: doc likes docs = ['aaabacdb abababdb addbaedb daecabdb badbccdb',
			'aeaaacdb abebabdb acdbaedc dbecadda addbbccb',
			'aeaaacdb abebabdb acdbaedc dbecadda addbbccb']
		The same is written on file.txt
		"""
		groupingAttr = "Trip_ID"
		grouped = data.groupby(groupingAttr)
		listDocs = []
		for name, group in grouped:
			#(TODO): remove prints before commit
			docWords = " ".join(group['Word'])
			#print "word doc",docWords
			listDocs.append(docWords)
		#print "documents list "
		#print listDocs
		return listDocs



def foo_partial():
	""" test for conversion cluster id to char """
	data = np.random.rand(100,1)
	print data
	worder = WordData(data)
	labels = worder.cluster_kmeans_signal(data,5)
	signalsChar = array('c')
	for l in labels:
		char = worder.label_to_char(l)
		signalsChar.append(char)
	l = signalsChar.tolist()
	l1 = signalsChar.tolist()
	### ElementWise operation to concatenate char in word
	newl =[m+n for m,n in zip(l,l1)]
	print newl

def foo_full():
	""" test full conversion """
	data = np.random.rand(100)
	df = pd.DataFrame(data= data)
	df[1] = pd.Series(data=data, index=df.index)
	worder = WordData(df)
	worder.create_words(worder.dataset)

def main():

	## import dataset precomputed
	## Not using method removing last column
	newData = pd.read_csv('../xsense_data/global_dataset.txt', sep=';')

	## Choose feature to represent in words
	## All exclused altitude
	dataPartOne = newData.ix[:,'Acc_X':'Pitch']
	dataPartTwo = newData.ix[:, 'Speed_X':'Speed_Z']

	newDataToWord = pd.concat([dataPartOne,dataPartTwo], axis=1)

	# print "list of words:",newDataToWord

	worder = WordData(newDataToWord)
	words = worder.create_words(worder.dataset)
	print words

	colWords = pd.Series(words, name='Word')
	wordDataset = pd.concat([newData,colWords], axis=1)
	wordDataset.to_csv('../xsense_data/word_global_dataset.txt',sep=';')

	docs = worder.create_text_corpus(wordDataset)


def main_new_dataset():
	""" Corpus creation method using Xsens dataset considering abs speed and diff yaw"""
	newData = pd.read_csv('../xsense_data/global_dataset_abs_speed_diff_yaw.txt', sep=';')

	listFeatures = ['Acc_X', 'Acc_Y', 'Speed_X', 'Speed_Y', 'Diff_Yaw']
	data = newData.ix[:, listFeatures]

	worder = WordData(data)
	words = worder.create_words(worder.dataset)
	print words
	print worder.discretization_model

	colWords = pd.Series(words, name='Word')
	wordDataset = pd.concat([newData,colWords], axis=1)
	wordDataset.to_csv('../xsense_data/word_global_dataset_abs_speed_diff_yaw.txt',sep=';')

if __name__ == "__main__":
	main_new_dataset()