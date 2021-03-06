import numpy as np
import pandas as pd
import os
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

MIN_START_PROB = 1e-200
MIN_TRANS_PROB = 1e-200
MIN_ALPHA_BETA = 1e-250
MIN_COV_VAL = 1e-5
MAX_ITER = 15
MIN_GINI = .5

class GaussianHMM(object):
	class RgznModes(object):
		def __init__(self):
			self.STANDARD = 0
			self.MAP = 1
			self.INERTIAL = 2
			self.MAP_SCALE_FREE = 3

	"""****************************"""
	"""       Initialization       """
	"""****************************"""

	def __init__(self, number_of_states, sequence, regularization_mode = 0):
		self.num_states = number_of_states
		emission_means, emission_covariances = self.get_dpmeans_emission_init(sequence)
		self.emission_density_objs = [multivariate_normal(mean=emission_means[k], cov=emission_covariances[k], allow_singular=True) for k in range(number_of_states)]
		start_probs, transition_probs = self.get_random_start_and_trans_probs()
		self.start_probs = start_probs
		self.trans_probs = transition_probs
		self.trans_update_method = [self.update_transition_probs_standard,
									self.update_transition_probs_MAP,
									self.update_transition_probs_inertial_regularized,
									self.update_transition_probs_MAP_scale_free,][regularization_mode]
		self.scaling_factors = None
		self.alpha_table = None
		self.beta_table = None
		self.gamma_table = None
		self.xi_table = None
		self.current_avg_ll = None
		self.log_likelihood = float('-inf')

	def get_random_start_and_trans_probs(self):
		K = self.num_states
		sub_table = np.random.random(size=(K,K))
		sub_table = (sub_table.T / sub_table.sum(axis=1)).T
		return sub_table[0], sub_table

	def get_dpmeans_emission_init(self, sequence):

		K = self.num_states
		data = pd.read_csv('../clustering/data_clustering/dp_means_12_Xsens.csv', sep=';')
		assignments = data['Cluster_Label'].values

		print assignments

		means = []
		covs = []
		for k in range(1,K+1):
			points = sequence[np.equal(assignments, k)]
			mean_vector = np.mean(points, axis=0)
			cov_matrix = np.cov(points, rowvar=0)
			cov_matrix[cov_matrix==0] = MIN_COV_VAL
			means.append(mean_vector)
			covs.append(cov_matrix)
		return means, covs

	def get_kmeans_emission_init(self, sequence):
		from sklearn import preprocessing

		K = self.num_states
		""" Added scaling for clustering """
		std_scale = preprocessing.StandardScaler().fit(sequence)
		sequence_scaled = std_scale.transform(sequence)

		assignments = KMeans(n_clusters=K).fit_predict(sequence_scaled)
		print assignments
		means = []
		covs = []
		for k in range(K):
			points = sequence[np.equal(assignments, k)]
			mean_vector = np.mean(points, axis=0)
			cov_matrix = np.cov(points, rowvar=0)
			cov_matrix[cov_matrix==0] = MIN_COV_VAL
			means.append(mean_vector)
			covs.append(cov_matrix)
		return means, covs

	"""****************************"""
	"""     Gaussian Emissions     """
	"""****************************"""

	def get_emission_prob(self, j, point):
		return self.emission_density_objs[j].pdf(point)

	def get_log_emission_prob(self, j, point):
		return self.emission_density_objs[j].logpdf(point)

	"""****************************"""
	"""     Forward / Backward     """
	"""****************************"""

	def forward(self, sequence):
		''' Implements the Forward Algorithm, creates alpha table. '''
		T = len(sequence)
		K = self.num_states
		x = sequence
		alpha_table = np.zeros((T, K), dtype="float64")
		scaling_factors = np.zeros(T)
		for k in range(K):
			alpha_table[0][k] = self.start_probs[k] * self.get_emission_prob(k, x[0])
		if not np.all(alpha_table[0]):
			for k in range(K):
				alpha_table[0][k] = self.start_probs[k] * self.get_emission_prob(k, x[0]) + MIN_ALPHA_BETA
		scaling_factors[0] = alpha_table[0].sum()
		alpha_table[0,:] /= scaling_factors[0]
		for t in range(1, T):
			for k in range(K):
				alpha_table[t][k] = np.dot(alpha_table[t-1,:], self.trans_probs[:,k]) * self.get_emission_prob(k, x[t])
			if not np.all(alpha_table[t]):
				for k in range(K):
					alpha_table[t][k] = np.dot(alpha_table[t-1,:], self.trans_probs[:,k]) * self.get_emission_prob(k, x[t]) + MIN_ALPHA_BETA
			scaling_factors[t] = alpha_table[t,:].sum()
			alpha_table[t,:] /= scaling_factors[t]
		''' save results '''
		self.alpha_table = alpha_table
		self.scaling_factors = scaling_factors

	def backward(self, sequence):
		''' Implements the Backward algorithm, creates beta table. '''
		T = len(sequence)
		K = self.num_states
		x = sequence
		beta_table = np.zeros((T, K), dtype="float64")
		beta_table[-1, :] = 1.
		for t in range(T-2, -1, -1):
			for k in range(K):
				beta_table[t][k] = np.sum([beta_table[t+1][j] * self.trans_probs[k][j] * self.get_emission_prob(j, x[t+1]) for j in range(K)])
			if not np.all(beta_table[t]):
				for k in range(K):
					beta_table[t][k] = np.sum([beta_table[t+1][j] * self.trans_probs[k][j] * self.get_emission_prob(j, x[t+1]) for j in range(K)]) + MIN_ALPHA_BETA
			beta_table[t,:] /= self.scaling_factors[t+1]
		''' save results '''
		self.beta_table = beta_table

	'''*****************************'''
	''' Compute Gamma and Xi Tables '''
	'''*****************************'''

	def compute_xi_table(self, sequence):
		T = len(sequence)
		K = self.num_states
		xi_table = np.zeros((T, K, K))
		for t in range(1, T):
			point = sequence[t]
			sf = self.scaling_factors[t]
			for i in range(K):
				for j in range(K):
					xi_table[t][i][j] = (1./sf) * self.alpha_table[t-1][i] * self.trans_probs[i][j] * self.get_emission_prob(j, point) * self.beta_table[t][j]
		self.xi_table = xi_table

	def compute_gamma_table(self, sequence):
		T = len(sequence)
		K = self.num_states
		gamma_table = np.zeros((T, K))
		for t in range(T):
			for k in range(K):
				gamma_table[t][k] = self.alpha_table[t][k] * self.beta_table[t][k]
		assert(self.alpha_table[-1].sum() == gamma_table[-1].sum())
		self.gamma_table = gamma_table
		self.log_likelihood = np.log(self.scaling_factors).sum()

	"""***************************"""
	""" Start Prob Update Methods """
	"""***************************"""

	def update_start_probs(self):
		self.start_probs = self.gamma_table[0,:] / np.sum([self.gamma_table[0][j] for j in range(self.num_states)])
		self.start_probs[self.start_probs < MIN_START_PROB] = MIN_START_PROB
		self.start_probs /= self.start_probs.sum()

	"""***************************"""
	""" Transition Update Methods """
	"""***************************"""

	def smooth_transition_probs(self):
		for k in range(self.num_states):
			self.trans_probs[k][self.trans_probs[k] < MIN_TRANS_PROB] = MIN_TRANS_PROB
			self.trans_probs[k] /= self.trans_probs[k].sum()

	def update_transition_probs_standard(self, sequence, dummy_param=None):
		T = len(sequence)
		K = self.num_states
		for j in range(K):
			denom = np.sum([self.xi_table[t][j][l] for l in range(K) for t in range(1, T)])
			for k in range(K):
				self.trans_probs[j][k] = np.sum([self.xi_table[t][j][k] for t in range(1, T)]) / denom
		self.smooth_transition_probs()

	def update_transition_probs_MAP(self, sequence, zeta=1):
		T = len(sequence)
		K = self.num_states
		for j in range(K):
			denom = (zeta - 1) + np.sum([self.xi_table[t][j][l] for l in range(K) for t in range(1, T)])
			for k in range(K):
				added_mass = (zeta - 1) if j == k else 0.
				self.trans_probs[j][k] = np.sum([added_mass + self.xi_table[t][j][k] for t in range(1, T)]) / denom
		self.smooth_transition_probs()

	def update_transition_probs_MAP_scale_free(self, sequence, zeta=1):
		T = len(sequence)
		K = self.num_states
		amp_val = ((T-1)**zeta - 1)
		for j in range(K):
			denom = np.sum([self.xi_table[t][j][l] for l in range(K) for t in range(1, T)]) + amp_val
			for k in range(K):
				added_mass = amp_val if j == k else 0.
				self.trans_probs[j][k] = np.sum([self.xi_table[t][j][k] + added_mass for t in range(1, T)]) / denom
		self.smooth_transition_probs()

	def update_transition_probs_inertial_regularized(self, sequence, zeta=1):
		T = len(sequence)
		K = self.num_states
		v = (T-1)**zeta
		amped_vals = [[None for k in range(K)]] + [[(self.gamma_table[t-1][k] - self.xi_table[t][k][k]) * v for k in range(K)] for t in range(1, T)]
		for j in range(K):
			denom = np.sum([self.xi_table[t][j][i] + (amped_vals[t][i] if i == j else 0.0) for i in range(K) for t in range(1, T)])
			for k in range(K):
				self.trans_probs[j][k] = np.sum([self.xi_table[t][j][k] + (amped_vals[t][k] if j == k else 0.0) for t in range(1, T)]) / denom
		self.smooth_transition_probs()

	"""************************"""
	""" Emission Update Method """
	"""************************"""

	def update_emission_parameters(self, sequence):
		T = len(sequence)
		K = self.num_states
		x = sequence
		D = x.shape[1]
		for k in range(K):
			denom = self.gamma_table[:,k].sum(axis=0)
			means = np.dot(self.gamma_table[:,k].T, x) / denom
			demeaned = (x - means)
			cov_mat = np.dot(self.gamma_table[:, k] * demeaned.T, demeaned) / denom + .01 * np.eye(D)
			self.emission_density_objs[k] = multivariate_normal(mean=means, cov=cov_mat, allow_singular=True)

	"""************************"""
	"""    Decode (Viterbi)    """
	"""************************"""

	def decode(self, sequence):
		''' Implements the Viterbi Algorithm '''
		T = len(sequence)
		K = self.num_states
		x = sequence
		V_table = np.zeros((T, K), dtype="float64")
		backpointers = np.zeros((T, K), dtype="float64")
		start_probs = self.start_probs[:]
		if not np.all(start_probs):
			start_probs += 1e-20
			start_probs /= start_probs.sum()
		for k in range(K):
			V_table[0][k] = self.get_log_emission_prob(k, x[0]) + np.log(start_probs[k])
		backpointers[0,:] = -1
		for t in range(1, T):
			for k in range(K):
				log_emiss_p = self.get_log_emission_prob(k, x[t])
				scores = [(V_table[t-1][j] + np.log(self.trans_probs[j][k]) + log_emiss_p,  j) for j in range(K)]
				V_table[t][k], backpointers[t][k] = max(scores)
		state_path = [np.max(backpointers[-1,:])]
		for t in range(backpointers.shape[0]-1, -1, -1):
			state_path.insert(0, backpointers[t][state_path[0]])
		return state_path[1:]

	"""************************"""
	"""          Learn         """
	"""************************"""

	def learn(self, sequence, zeta=None, init=False, only_final_ll=False):
		''' Runs Baum-Welch to train HMM '''
		K = self.num_states
		epsilon = 0.1
		old_ll = float("-inf")
		iterations = 0
		if init:
			means, covs = self.get_kmeans_emission_init(sequence)
			self.emission_density_objs = [multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True) for k in range(K)]
			start_probs, transition_probs = self.get_random_start_and_trans_probs()
			self.start_probs = start_probs
			self.trans_probs = transition_probs
		while True:
			iterations += 1
			self.forward(sequence)
			self.backward(sequence)
			self.compute_gamma_table(sequence)
			self.compute_xi_table(sequence)
			if not only_final_ll:
				print "Log Likelihood:", self.log_likelihood
			if abs(self.log_likelihood - old_ll) < epsilon or iterations > MAX_ITER:
				print "Final Log Likelihood:", self.log_likelihood
				return
			else:
				old_ll = self.log_likelihood
			self.update_start_probs()
			self.trans_update_method(sequence, zeta)
			self.update_emission_parameters(sequence)

	def get_segments(self, state_assignments):
		segments = []
		state = state_assignments[0]
		start = 0
		count = 0
		for i in range(len(state_assignments)):
			if state_assignments[i] == state:
				count += 1
			else:
				segments.append(((start, start+count-1), state))
				start = i
				count = 1
				state = state_assignments[i]
		segments.append(((start, start+count-1), state))
		return segments

	def learn_param_free(self, sequence, param_range=[1,10], only_final_ll=False):
		epsilon = 0.01
		p = np.mean(param_range)
		print "Parameter:", p
		self.learn(sequence, zeta=p, init=True, only_final_ll=only_final_ll)
		hidden_states = self.decode(sequence)
		segments = self.get_segments(hidden_states)


def main():
	from data_import.ImportDataset import ImportData

	""" Import dataset """
	data = pd.read_csv('../xsense_data/global_dataset_abs_speed_diff_yaw.txt', sep=';')
	#Passare una matrix contenente solamente le colonne necessarie al clustering e quelle per HMM
	data_model = data[['Acc_X','Acc_Y','Speed_X','Speed_Y','Diff_Yaw']].as_matrix()


	""" Regularization modes and parameter """
	rgzn_modes = GaussianHMM.RgznModes()
	zeta = 3.

	""" Define two-state model, run it on observation data and find
		maximally likely hidden states, subject to regularization. """
	K = 5

	model5_free = GaussianHMM(K,data_model, rgzn_modes.STANDARD)
	model5_free.learn(data_model, zeta= zeta)
	predicted_states_free5 = model5_free.decode(data_model)

	print "Predicted states free", predicted_states_free5
	print "Transition probabilities free", model5_free.trans_probs

	model5_inertial = GaussianHMM(K, data_model, rgzn_modes.INERTIAL)
	model5_inertial.learn(data_model, zeta=zeta)
	predicted_states_inertial5 = np.array(model5_inertial.decode(data_model))

	print "Predicted states inertial", predicted_states_inertial5
	print "Transition probabilities inertial", model5_inertial.trans_probs

	model5_map = GaussianHMM(K, data_model, rgzn_modes.MAP)
	model5_map.learn(data_model, zeta=zeta)
	predicted_states_map5 = np.array(model5_map.decode(data_model))

	print "Predicted states map", predicted_states_map5
	print "Transition probabilities map", model5_map.trans_probs


	"""Comparison free and inertial K = 7"""
	K = 7
	model7_inertial = GaussianHMM(K,data_model, rgzn_modes.INERTIAL)
	model7_inertial.learn(data_model, zeta=zeta)
	predicted_states_inertial7 = model7_inertial.decode(data_model)

	print "Predicted states", predicted_states_inertial7
	print "Transition probabilities ", model7_inertial.trans_probs

	"""Comparison free and inertial K = 10"""
	K = 10
	model10_inertial = GaussianHMM(K, data_model, rgzn_modes.INERTIAL)
	model10_inertial.learn(data_model, zeta=zeta)
	predicted_states_inertial10 = model10_inertial.decode(data_model)

	print "Predicted states", predicted_states_inertial10
	print "Transition probabilities ", model10_inertial.trans_probs


	model10_free = GaussianHMM(K, data_model, rgzn_modes.STANDARD)
	model10_free.learn(data_model, zeta= zeta)
	predicted_states_free10 = model10_free.decode(data_model)

	print "Predicted states", predicted_states_free10
	print "Transition probabilities ", model5_free.trans_probs

def save_csv_hmm_results(data, file_name):
	"""
	:param data: dataframe to be written, last column contains clustering labels
	:param file_name: name of the file
	:return: null
	"""
	format = '.csv'
	dir_name = 'data_hmm'
	full_path = os.path.join(dir_name, file_name + format)
	data.to_csv(full_path, sep=';', index=False)
	#data.to_csv(file_name, sep=';', index=False)

def run_dp12_init():
	data = pd.read_csv('../xsense_data/global_dataset_abs_speed_diff_yaw.txt', sep=';')
	#Passare una matrix contenente solamente le colonne necessarie al clustering e quelle per HMM
	data_model_df =  data[['Acc_X','Acc_Y','Speed_X','Speed_Y','Diff_Yaw']]
	data_model = data_model_df.as_matrix()

	""" Regularization modes and parameter """
	rgzn_modes = GaussianHMM.RgznModes()
	zeta = 3.

	""" Define two-state model, run it on observation data and find
		maximally likely hidden states"""
	K = 7

	model7_free = GaussianHMM(K,data_model, rgzn_modes.STANDARD)
	model7_free.learn(data_model, zeta= zeta)
	predicted_states_free7 = model7_free.decode(data_model)
	emission_obj = model7_free.emission_density_objs

	print "Predicted states inertial", predicted_states_free7
	print "Transition probabilities inertial", np.around(model7_free.trans_probs, decimals=4)
	for i in range(K):
		print "Emission Object", i, " : ", emission_obj[i].mean

	data_model_df.loc[:, 'Cluster_Label'] = pd.Series(predicted_states_free7, index=data_model_df.index)
	data_model_df.loc[:, 'Trip_ID'] = data[['Trip_ID']]

	save_csv_hmm_results(data_model_df, "hmm_inertial_7_XSens")

def run_inertial_dp12_init():

	data = pd.read_csv('../xsense_data/global_dataset_abs_speed_diff_yaw.txt', sep=';')
	#Passare una matrix contenente solamente le colonne necessarie al clustering e quelle per HMM
	data_model_df =  data[['Acc_X','Acc_Y','Speed_X','Speed_Y','Diff_Yaw']]
	data_model = data_model_df.as_matrix()

	""" Regularization modes and parameter """
	rgzn_modes = GaussianHMM.RgznModes()
	zeta = 3.

	""" Define two-state model, run it on observation data and find
		maximally likely hidden states"""
	K = 7

	model7_inertial = GaussianHMM(K, data_model, rgzn_modes.INERTIAL)
	model7_inertial.learn(data_model, zeta=zeta)
	predicted_states_inertial7 = model7_inertial.decode(data_model)
	emission_obj = model7_inertial.emission_density_objs

	print "Predicted states free", predicted_states_inertial7
	print "Transition probabilities free", np.around(model7_inertial.trans_probs, decimals=4)
	for i in range(K):
		print "Emission Object", i, " : ", emission_obj[i].mean

	data_model_df.loc[:, 'Cluster_Label'] = pd.Series(predicted_states_inertial7, index=data_model_df.index)
	data_model_df.loc[:, 'Trip_ID'] = data[['Trip_ID']]

	save_csv_hmm_results(data_model_df, "hmm_free_7_XSens")

if __name__ == "__main__":
	#main()
	run_inertial_dp12_init()