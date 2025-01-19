# Solution Description
The main aim of the project is implementing a Naive Bayes Classifier, a model for classifying datasets with different types of features. 
- MultinomialNaiveBayesClassifier was implemented to classify datasets with categorical features (Mushrooms dataset).
- GaussianNaiveBayesClassifier was implemented to support classifying of data with continuous features (Iris dataset).
  
## 1. Class: MultinomialNaiveBayesClassifier

### Training (fit method)
For each class C count prior probability (stored in self.class_probs):  
    P(C) = (number of samples in C)/(total samples)
    
For each class C and feature i:  
   - Extract all samples belonging to the class (class_data)  
   - Count the occurrences of each unique value of feature i (counts)  
   - Count conditional probabilities using Laplace smoothing (self.feature_probs)  

### Prediction (predict method)
#### 2.1 Log-Probability Scores log(P(C|X)) (_calculate_log_scores)

For each sample in X:  
  - Initialize the score with the log prior log(P(C))  
  - For each feature i add P(x(i)|C)  
  - Store scores for all classes in log_scores  
#### 2.2 Class Prediction  
For each sample choose the class with the highest log(P(C|X))  

### Predicting Probabilities (predict_proba method)  
- Call _calculate_log_scores  
- Exponentiate each log-score  
- Normalize probabilities   
- Store probabilities in a dictionary for each sample  

## 2. Gaussian Naive Bayes Classifier

### Training (fit method)
For each class C count prior probability:
    P(C) = (number of samples in C)/(total samples)
    
For each feature i in class C:
   - Count the mean of the feature for all samples of the class
   - Count the standard deviation for feature i for all samples of the class 
   - Count prior probability for the class
     
Store the means, standard deviations, and prior probabilities for each class

### Prediction
#### 2.1 Log-Probability Scores log(P(C|X))
For each sample in X:
  For each class C:
    - count the log prior
    - count the log likelihood using the Gaussian distribution formula
    - count the log-posterior probability (sum log prior and log likelihoods)
#### 2.2 Class Prediction
For each sample choose the class with the highest log-posterior probability.

### Predicting Probabilities (predict_proba method)
- Call _compute_posteriors to compute log-posterior probabilities for each class.
- Exponentiate the log-posterior probabilities.
- Normalize the resulting probabilities to ensure they sum to 1.
- Return the normalized class probabilities for each sample.
