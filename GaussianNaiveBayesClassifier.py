import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from math import pi
class GaussianNaiveBayes:
    
    def __init__(self):
        self.means = None
        self.stds = None # standard deviation
        self.priors = None # prior probabilties
        self.classes = None

    def fit(self, X, y):
        # count probabilities for each class and value by training data
        self.classes = np.unique(y) # unique class labels
        class_num = len(self.classes)
        f_num = X.shape[1]
        # mean, standard deviation, and prior arrays
        self.means = np.zeros((class_num, f_num))
        self.priors = np.zeros(class_num)
        self.stds = np.zeros((class_num, f_num))
        # statistics for each class
        for i in range(class_num):
            class_data = X[y == self.classes[i]]
            self.means[i, :] = np.mean(class_data, axis=0)
            self.stds[i, :] = np.std(class_data, axis=0) + 1e-6  #avoid division by zero
            self.priors[i] = class_data.shape[0] / X.shape[0]

    def predict(self, X):
        # predict the most likely class for each sample
        posteriors = self._compute_posteriors(X) # posterior probabilities
        return self.classes[np.argmax(posteriors, axis=1)]

    def predict_proba(self, X):
        # count probability of each class for every sample
        posteriors = self._compute_posteriors(X)
        prob = np.exp(posteriors - np.max(posteriors))
        return prob / np.sum(prob)

    def _compute_posteriors(self, X):
        
        sample_num = X.shape[0]
        class_num = len(self.classes)
        posteriors = np.zeros((sample_num, class_num))

        for i in range(class_num):
            mean = self.means[i]
            std = self.stds[i]

            log_prior = np.log(self.priors[i])
            log_likelihood = np.sum(((mean - X) ** 2) / ((-2) * (std ** 2)), axis=1)
            log_likelihood -= np.sum(np.log(2 * pi * (std ** 2))) / 2

            posteriors[:, i] = log_prior + log_likelihood

        return posteriors
    
def run_multiple_times(gnb, X, y, iterations=10, test_size=0.3, random_state=1):
    # runs code multiple times and count average accuracy
    acc_list = []
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state + i)
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        acc_list.append(accuracy)
    return np.mean(acc_list)

iris = datasets.load_iris()
gnb = GaussianNaiveBayes()
X,y = iris.data, iris.target

av_acc = run_multiple_times(gnb, X, y, iterations=100) # average accurancy
print(f'Average Accuracy: {av_acc:.3f}')
