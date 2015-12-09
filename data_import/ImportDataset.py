import pandas as pd

import datetime

__author__ = 'Paolo'

""" Class used for import datasets """


class ImportData:
	def __init__(self, path):
		self.dataPath = path

	def import_csv(self):
		data = pd.read_csv('C:/Users/Paolo/Desktop/Reply/Thesis/Data/XsenseData/MT_07700161-003-001.txt', sep=';')
		data = data.drop(data.columns[[19]], axis=1)
		#print data
		return data

	def import_csv_from_path(self, path):
		data = pd.read_csv(path, sep=';')
		data = data.drop(data.columns[[19]], axis=1)
		return data

	def import_csv_reduced(self):
		data = pd.read_csv(self.dataPath, sep=';')
		# Remove last column containing empty value
		data = data.drop(data.columns[[len(data.columns)-1]], axis=1)
		data = data.head(10000)
		return data

	def get_clustering_table(self, data):
		# Passare una matrix contenente solamente le colonne necessarie al clustering e quelle per HMM
		dataNew = data.ix[:, 'FreeAcc_X':].as_matrix()
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
	data = imp.import_csv()

	""" Plotting AccX """
	y = data.ix[:, 'FreeAcc_X'].values
	x = np.array(xrange(len(y)))
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	ax1.plot(x,y)
	fig1.show()

	""" Plotting AccY """
	y = data.ix[:, 'FreeAcc_Y'].values
	ax2 = fig1.add_subplot(111)
	ax2.plot(x,y)
	plt.show()





