import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from array import *
from sklearn.metrics import silhouette_samples, silhouette_score

__author__ = 'Paolo'

""" Class used to create text corpus """

class WordData():
	def __init__(self, data):
		self.dataset = data
		self.models = {}
		## max 10 cluster as assumption
		self.dictionary = ["a","b","c","d","e","f","g","h","i","j"]

	def cluster_kmeans_signal(self, signal, nClusters, featureName):
		"""
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
		for column in dataset.columns:
			# print "column", column
			series = dataset[column].as_matrix()
			#(TODO): Remove prints
			# print "valori series:" , dataset[column]
			# print "valori matrice:", series

			labels = self.cluster_kmeans_signal(series.reshape(-1, 1),5, column)
			# print labels
			if dataset.columns.get_loc(column) == 0:
				newl = self.label_to_char(labels)
			else:
				charLabels = self.label_to_char(labels)
				newl =[m+n for m,n in zip(newl,charLabels)]
		# print newl
		return newl

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

	# from data_import.ImportDataset import ImportData

	# imp = ImportData('C:/Users/Paolo/Desktop/Reply/Thesis/Data/XsenseData/MT_07700161-003-001.txt')

	# newData = imp.import_all_files()

	## import dataset precomputed
	## Not using method removing last column
	newData = pd.read_csv('../xsense_data/global_dataset.txt', sep=';')
	tripId = newData.ix[:,'Trip_ID']

	## Choose feature to represent in words
	## All exclused altitude
	dataPartOne = newData.ix[:,'Acc_X':'Pitch']
	dataPartTwo = newData.ix[:, 'Speed_X':'Speed_Z']

	#newDataToWord.insert(0,"Trip_ID",tripId)

	newDataToWord = pd.concat([dataPartOne,dataPartTwo], axis=1)

	# print "list of words:",newDataToWord

	worder = WordData(newDataToWord)
	words = worder.create_words(worder.dataset)
	print words


if __name__ == "__main__":
	main()