import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')

__author__ = 'Paolo'

class SignalsClustering():
	def __init__(self):
		self.data = pd.read_csv('../xsense_data/global_dataset.txt', sep=';')

	def signal_thresholds(self,signal, n_bins):
		"""
		:param signal: Series object to be processed
			n_bin : number of bins
		:return: left_edges_thresholds : is a list of the left thresholds used for binning
		"""
		counts, bins = np.histogram(signal, bins=n_bins)
		#left_edges_thresholds = np.delete(bins,[0,n_bins])
		left_edges_thresholds = bins
		return left_edges_thresholds

	def assign(self,signal,thresholds):
		"""
		:param signal: Series object to be processed
			thresholds: array of thresholds used for discretization process
		:return: assign_labels : is an array of char labsels
		"""
		assign_labels = np.zeros(len(signal), dtype=np.int)
		for i in range(1,len(thresholds)):
			index_ass = [l for l in range(len(signal)) if (signal[l] <= thresholds[i]) & (signal[l] >= thresholds[i-1])]
			assign_labels[index_ass] = i-1

		return assign_labels

	def plotCharts(self):
		acc_X = self.data.ix[:,'Acc_X']
		acc_Y = self.data.ix[:,'Acc_Y']
		acc_Z = self.data.ix[:,'Acc_Z']
		speed_X = self.data.ix[:,'Speed_X']
		roll = self.data.ix[:,'Roll']


		""" Bins left edges computation """
		counts, bins = np.histogram(acc_X, bins= 5)
		print "bins: ", bins
		print "max", max(acc_X)
		print "min", min(acc_X)
		acc_X_bins_left = pd.Series(counts, index=bins[:-1])
		print acc_X_bins_left
		accX_bins_leftedges =  np.delete(bins,[0,5])
		print "left edges", accX_bins_leftedges


		"""Histograms signals """
		df = pd.concat([acc_X,acc_Y,acc_Z,speed_X,roll], axis=1)
		df.diff().hist(bins=50)
		plt.show()

		"""Box Plot """
		dAcc = df.ix[:,['Acc_X','Acc_Y','Acc_Z']]
		dAcc.boxplot(showmeans=True)
		plt.show()


	def normalize(self, data):
		dataValues = data.values.astype(float)

		# Create a minimum and maximum processor object
		minMaxScaler = preprocessing.MinMaxScaler()

		# Create an object to transform the data to fit minmax processor
		dataScaled = minMaxScaler.fit_transform(dataValues)

		# Run the normalizer on the dataframe
		normalized = pd.DataFrame(dataScaled)

		return normalized

	def clusterSignal(self):
		acc_XNormalized = self.normalize(self.data['Acc_X'])
		km = KMeans(n_clusters=5, init='k-means++')
		"""Binning Acc_X """
		acc_XNormalizedValues = acc_XNormalized.as_matrix()
		#print acc_XNormalizedValues
		km.fit(acc_XNormalizedValues.reshape(-1,1))
		km.predict(acc_XNormalizedValues.reshape(-1,1))
		#print km.labels_
		#for p in km.labels_: print p
		print km.cluster_centers_
		#silhouette_avg = silhouette_score(acc_XNormalizedValues, km.labels_)
		#print silhouette_avg


def main():
	o = SignalsClustering()
	#o.plotCharts()

	""" Test discretization """
	signal = pd.Series([0.1,0.2,0.3,0.3,0.4,0.4,0.5,0.5,1,1,2,2,2,3,7,11,12,13,1,8,25], name="test_Series")
	t = o.signal_thresholds(signal,5)
	assignments = o.assign(signal, t)
	print "assignments", assignments
	print "thresholds", t
	print "Series name", signal.name


if __name__ == "__main__":
	main()