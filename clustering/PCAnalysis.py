import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

__author__ = 'Paolo'

def pca_analysis(path, features):
	df_data = pd.read_csv(path, sep=';')
	X = df_data[features]
	labels = df_data

if __name__ == "__main__":
	pca_analysis()