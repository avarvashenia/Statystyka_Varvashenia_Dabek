import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:
	def __init__(self):
		self.class_probs = {}
		self.feature_probs = defaultdict(dict) 
	
	def _calculate_log_scores(self, X):
		# counts the log-probability scores for each class for each sample in X

		log_scores_list = [] # will store log-probability scores for all samples
		
		for sample in X:
			log_scores = {} # store log-probability score for current sample
			# iterate over each class and its prior probability
			for clss, class_prob in self.class_probs.items():
				score = np.log(class_prob)
				# iterate over each feature in the sample
				for i in range(len(sample)):
					feature_value = sample[i] # current feature value
					feature_prob = self.feature_probs[clss].get(i, {}).get(feature_value, 1e-6)
					# probability of feature_value for the current class
					# + assigning small default probability to aviod zero values
					score += np.log(feature_prob)
				log_scores[clss] = score # final log-probability score for class
			log_scores_list.append(log_scores) # # add the scores for the current sample to list
		return log_scores_list
	
	def fit(self, X, y):
		# counts probabilities for each class and value by training data
		unique_classes = np.unique(y) # [0 1] - edible and poisonous
		
		for clss in unique_classes:
			class_num = np.sum(clss==y)
			probability = class_num / len(y)
			self.class_probs[clss] = probability # {0: 0.4815, 1: 0.5185} - probability of class in dataset
			# (percentage of data belonging to class)
		
		features_num = X.shape[1] # 23 - 'class' - 'stalk-root' = 21 - number of features in dataset

		for clss in self.class_probs:
			class_indices = np.where(y == clss)[0]  # indices of samples that belong to the current class
			class_data = X[class_indices] # subset of feature matrix X where samples belong to the current class
			
			for i in range(features_num):
				values, counts = np.unique(class_data[:, i], return_counts=True)
				# values = unique values of the current feature
				# counts = number of each unique value
				total = len(class_data)
				values_len = len(values)
				feature_prob_dict = {} # will store the probability of each value for the current feature
				# for each unique value of the current feature calculate the probability
				for j in range(len(values)):
					value = values[j]
					count = counts[j]
					feature_prob_dict[value] = (count + 1) / (total + values_len)
				# probability was calculated using Laplace smoothing to avoid zero probabilities for unseen values
				
				self.feature_probs[clss][i] = feature_prob_dict # probability dictionary for feature i in the class
	
	def predict(self, X):
		# predict the most likely class for each sample (using probabilities from fit method)
		log_scores_list = self._calculate_log_scores(X) # log-probability scores for all samples
		predictions = [] # # list for storing probability distributions for each sample
		# Iterate over log-probability scores for each sample
		for scores in log_scores_list:
			best = max(scores, key=scores.get) # Class with the highest log-probability score
			predictions.append(best) # Add the predicted class to the list
		
		return np.array(predictions)
	
	def predict_proba(self, X):
		# counts probability of each class for every sample in the input feature matrix

		log_scores_list = self._calculate_log_scores(X) # # Log-probability scores for all samples
		probabilities = [] # probability distributions
		
		for log_scores in log_scores_list:
			max_score = max(log_scores.values())
			exp_scores = {} # exponentiated scores for the current sample
			# calculate exponentiated scores for each class
			for clss, score in log_scores.items():
				exp_scores[clss] = np.exp(score - max_score)
			total_score = sum(exp_scores.values())
			
			probabilities.append({})
			# calculate normalized probabilities for each class
			for clss, exp_score in exp_scores.items():
				probabilities[-1][clss] = exp_score / total_score
		return probabilities
	
	def accuracy(self, X, y):
		predictions = self.predict(X)
		return np.mean(predictions == y)

def run_multiple_tests(nb, X, y, iterations=10, test_size=0.3, random_state=1):
	acc_list = []
	for i in range(iterations):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state + i)
		nb.fit(X_train, y_train)
		accuracy = nb.accuracy(X_test, y_test)
		acc_list.append(accuracy)
	
	return np.mean(acc_list)

file_path = 'archive/mushrooms.csv'
df = pd.read_csv(file_path)
df = df.drop(columns=['stalk-root'])
for column in df.columns:
	df[column] = pd.factorize(df[column])[0]

X = df.drop('class', axis=1).values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

nb = NaiveBayesClassifier()
avg_accuracy = run_multiple_tests(nb, X, y, iterations=100)
print(f"Average Accuracy: {avg_accuracy:.3f}")
