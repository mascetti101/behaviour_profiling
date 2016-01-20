import pandas as pd

class DSDProcess():
	def __init__(self):
		self.data = pd.read_csv('../xsense_data/global_dataset.txt', sep=';')
		self.thresholds = {}
	def risk_acceleration_percentage(self, data):
		risk_data = data.query('Acc_X >')
		risk_percentage = len(risk_data.index)/len(data.index)
		return risk_percentage
