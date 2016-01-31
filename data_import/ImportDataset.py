import pandas as pd
import datetime
from DatasetBuilder import DatasetBuilder
from os import listdir
from os.path import isfile, join

__author__ = 'Paolo'

""" Class used for import datasets """


class ImportData:
	#(TODO): modify initializer
	def __init__(self, path):
		self.dataPath = path
		self.dirPath = "C:/Users/Paolo/Desktop/Reply/Thesis/Data/XsenseData/Data/"
		self.dirPath_new = "C:/Users/Paolo/Desktop/Reply/Thesis/Data/hcilabData/"

	def import_all_files(self):
		"""
		:return: aggregated dataset representing all collected trips as a Dataframe
		"""
		listFiles = [f for f in listdir(self.dirPath) if isfile(join(self.dirPath, f))]
		fileIndex = 0
		globalData = pd.DataFrame();
		for f in listFiles:
			fileIndex += 1
			data = self.import_csv_from_path(self.dirPath+f)
			dBuilder = DatasetBuilder()

			#### Aggregate trip dataset
			newData = dBuilder.computeDataset(data)

			#(TODO): think about dataset normalization

			values = [fileIndex] * len(newData.index)
			tripCol = pd.Series(values, index=newData.index.values)
			newData.insert(0,"Trip_ID",tripCol)
			globalData = pd.concat([globalData,newData])
			#(TODO): Remove print

		globalData.to_csv('../xsense_data/global_dataset.txt',sep=';')
		return globalData


	def import_all_files_new_dataset(self):
		"""
		:return: aggregated dataset representing all collected trips as a Dataframe
		"""
		listFiles = [f for f in listdir(self.dirPath_new) if isfile(join(self.dirPath_new, f))]
		fileIndex = 0
		globalData = pd.DataFrame();
		for f in listFiles:
			fileIndex += 1
			data = pd.read_csv(self.dirPath_new + f, sep=';')
			dBuilder = DatasetBuilder()

			#### Aggregate trip dataset
			newData = dBuilder.computeDataset_hcilab(data)

			#(TODO): think about dataset normalization

			values = [fileIndex] * len(newData.index)
			tripCol = pd.Series(values, index=newData.index.values)
			newData.insert(0,"Trip_ID",tripCol)
			globalData = pd.concat([globalData,newData])
			#(TODO): Remove print

		globalData.to_csv('../xsense_data/global_dataset_hcilab.txt',sep=';')
		return globalData


	#(TODO): Remove this useless method , used by path
	def import_csv(self):
		data = pd.read_csv('C:/Users/Paolo/Desktop/Reply/Thesis/Data/XsenseData/MT_07700161-003-001.txt', sep=';')
		data = data.drop(data.columns[[len(data.columns)-1]], axis=1)
		#print data
		return data

	def import_csv_from_path(self, path):
		data = pd.read_csv(path, sep=';')
		# Remove last column containing empty value
		data = data.drop(data.columns[[len(data.columns)-1]], axis=1)
		return data

	""" Import head file just for testing purposes """
	def import_csv_reduced(self):
		data = pd.read_csv(self.dataPath, sep=';')
		# Remove last column containing empty value
		data = data.drop(data.columns[[len(data.columns)-1]], axis=1)
		data = data.head(10000)
		return data

	def get_clustering_table(self, data):
		# Passare una matrix contenente solamente le colonne necessarie al clustering e quelle per HMM
		dataNew = data.ix[:, 'Acc_X':].as_matrix()
		return dataNew

	def build_date(self, data):
		# data passato come array
		date = datetime.datetime(data['UTC_Year'], data['UTC_Month'], data['UTC_Day'], data['UTC_Hour'],
								 data['UTC_Minute'], data['UTC_Second'])
		return date


if __name__ == '__main__':

	import numpy as np
	import matplotlib.pyplot as plt

	# Test class functionalities
	print "Executing import dataset"
	imp = ImportData('C:/Users/Paolo/Desktop/Reply/Thesis/Data/XsenseData/MT_07700161-003-001.txt')
	# data = imp.import_csv()
	#
	""" Import all files """
	imp.import_all_files_new_dataset()
	#
	# """ Plotting AccX """
	# y = data.ix[:, 'FreeAcc_X'].values
	# x = np.array(xrange(len(y)))
	# fig1 = plt.figure()
	# ax1 = fig1.add_subplot(111)
	# ax1.plot(x,y)
	# fig1.show()
	#
	# """ Plotting AccY """
	# y = data.ix[:, 'FreeAcc_Y'].values
	# ax2 = fig1.add_subplot(111)
	# ax2.plot(x,y)
	# plt.show()





