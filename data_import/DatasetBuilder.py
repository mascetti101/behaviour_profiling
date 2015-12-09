import pandas as pd
import numpy as np

__author__ = 'Paolo'

""" Class used to perform dataset transformation and aggregation """
""" such as data aggregation and new features computation  		 """

class DatasetBuilder:
	def __init__(self):
		pass

	def computeDataset(self, data):
		groupingAttr = ['UTC_Year','UTC_Month', 'UTC_Day','UTC_Hour','UTC_Minute','UTC_Second']
		grouped = data.groupby(groupingAttr)

		""" Latitude, Longitude aggregated  """
		gLat = pd.Series(grouped['Latitude'].median(), name='Latitude')
		gLong = pd.Series(grouped['Longitude'].median(), name='Longitude')

		""" Aggregated values AccX """
		meanAccX = pd.Series(grouped['FreeAcc_X'].mean(), name='Acc_X')
		maxAccX = pd.Series(grouped['FreeAcc_X'].min(), name='Max_Acc_X')
		minAccX = pd.Series(grouped['FreeAcc_X'].max(), name='Min_Acc_X')

		""" Aggregated values AccY """
		meanAccY = pd.Series(grouped['FreeAcc_Y'].mean(), name='Acc_Y')
		maxAccY = pd.Series(grouped['FreeAcc_Y'].min(), name='Max_Acc_Y')
		minAccY = pd.Series(grouped['FreeAcc_Y'].max(), name='Min_Acc_Y')

		""" Aggregated values AccZ """
		meanAccZ = pd.Series(grouped['FreeAcc_Z'].mean(), name='Acc_Z')
		maxAccZ = pd.Series(grouped['FreeAcc_Z'].min(), name='Max_Acc_Z')
		minAccZ = pd.Series(grouped['FreeAcc_Z'].max(), name='Min_Acc_Z')

		""" Aggregated values Yaw,Roll,Pitch """
		meanYaw = pd.Series(grouped['Yaw'].mean(), name='Yaw')
		meanRoll = pd.Series(grouped['Roll'].mean(), name='Roll')
		meanPitch = pd.Series(grouped['Pitch'].mean(), name='Pitch')

		""" Aggregated values Altitude """
		meanAlt = pd.Series(grouped['Altitude'].mean(), name='Altitude')

		""" Aggregated values SpeedX,SpeedY,SpeedZ """
		meanSpeedX = pd.Series(grouped['Vel_X'].mean(), name='Speed_X')
		meanSpeedY = pd.Series(grouped['Vel_Y'].mean(), name='Speed_Y')
		meanSpeedZ = pd.Series(grouped['Vel_Z'].mean(), name='Speed_Z')

		newDataset = pd.concat([gLat, gLong, meanAccX,maxAccX,minAccX,meanAccY,maxAccY,
								minAccY,meanAccZ,maxAccZ, minAccZ, meanYaw, meanRoll,
								meanPitch, meanAlt, meanSpeedX, meanSpeedY, meanSpeedZ ], axis=1)
		return newDataset

	def normalize(self, data):
		normalized_data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
		return normalized_data

def main():
	from data_import.ImportDataset import ImportData
	importer = ImportData('C:/Users/Paolo/Desktop/Reply/Thesis/Data/XsenseData/Data/2015-12-01/BNC-021G/Section 1/Xsens_2015-12-01_11-12_BNC-021G_Section1-000.txt')
	data = importer.import_csv_reduced()
	dBuilder = DatasetBuilder()
	newData = dBuilder.computeDataset(data)
	normNewData = dBuilder.normalize(newData)
	normNewData.reset_index(level=0, inplace=True)
	print normNewData

if __name__ == '__main__':
	main()