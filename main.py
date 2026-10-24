from scipy.io import arff
import pandas as pd
import numpy as np
from scipy.optimize import minimize



# Load the ARFF file into a pandas dataframe
data, meta = arff.loadarff('diabetes.arff')
df = pd.DataFrame(data)
df['class'] = df['class'].str.decode('utf-8')

# see what the data table looks like
print(df)

# split into predictors X and target Y
X = df.drop('class', axis=1)
Y = df['class']



# write a function that applies a specified linear decision threshold to the entire dataset
# and returns the predictions of diabetes risk
def linear_decision(w: np.array) -> np.array:
    """
    apply a linear decision threshold to the predictors X
    :param w: array of weights and bias. Should be length-9: [w1, w2... w8, b]
    :return: numpy array of predictions for each case in X
    """

    linear_decision_threshold = X @ w[0:8] + w[8]

    at_risk = linear_decision_threshold > 0

    return np.array(at_risk)
    # your code here

# param w: one of the split groups
# Formula: H(V) = -Σk *P(vk)log2P(vk) from slide 18
def calc_entropy(group: np.array) -> float:
    positives = np.sum(group == 'tested_positive')
    negatives = np.sum(group == 'tested_negative')
    total = len(group)

    if ((positives == 0) or (negatives == 0)): #so log(0) doesnt break anything
        return 0
    
    return -((positives/total)*np.log2(positives/total) + (negatives/total)*np.log2(negatives/total))

# write a function that computes the entropy of patient classes in the two groups created by the linear decision threshold
def decision_entropy(w: np.array) -> float:
    """
    compute the average entropy of the 2 patient groups created by the linear decision threshold
    :param w: array of weights and bias. Should be length-9: [w1, w2... w8, b]
    :return: entropy in the actual classes (as contained in Y) of patients in each group
            (remember when computing average entropy, to weight by the size of each group - see slide 19)
    """
    # your code here
    at_risk = linear_decision(w)
    group_at_risk = Y[at_risk]
    group_no_risk = Y[~at_risk]

    n_at_risk = len(group_at_risk)
    n_no_risk = len(group_no_risk)
    n_total = n_at_risk + n_no_risk 
    return (n_at_risk/n_total*calc_entropy(group_at_risk)) + (n_no_risk/n_total*calc_entropy(group_no_risk))

# write a search that finds the best (entropy-minimizing) weights and bias terms
# YOUR CODE HERE
weights_and_bias = [1,2,3,4,5,-6,-7, -8,0.5]
# print(linear_decision(weights_and_bias))
print(f'Entropy with arbitrary weights and bias: {decision_entropy(weights_and_bias)}')

baseline = calc_entropy(Y)
print(f'Baseline entropy: {baseline}')
result = minimize(decision_entropy, x0=weights_and_bias, method='Powell')
print(f'Best entropy found: {result.fun}')
print(f'Best x values found: {result.x}')

# ___Reflection questions:___

# 1) What was the baseline entropy of the whole dataset? What is the entropy after your classifier splits it into two groups?


# 2) what effect does b have in the classification?
# If b is increased, there's a higher chance for the patient to be classified as being at risk
# while if b is decreased, there's a higher change for the patient to be classified as being not at risk.

# 3) what does the model tell us about the relationship between the clinical measurements and diabetes? Which measurements contibute to risk of diabetes, and which reduce it?
# In this model, a positive measurement means an increase in diabetes risk, while a negative measurement means a decrease in diabetes risk.
# 
#     1. Number of times pregnant
#           Best value: -34.28481442 so it strongly decreases risk of diabetes
#     2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test: 
#           Best value: 2.97328449 so it increases risk of diabetes
#     3. Diastolic blood pressure (mm Hg)
#           Best value: 3.02316677 so it increases risk of diabetes
#     4. Triceps skin fold thickness (mm)
#           Best value: 3.86057954 so it increases risk of diabetes
#     5. 2-Hour serum insulin (mu U/ml)
#           Best value: 10.14589807 so it strongly increases risk of diabetes
#     6. Body mass index (weight in kg/(height in m)^2)
#           Best value: -6.04523307 so it decreases risk of diabetes
#     7. Diabetes pedigree function
#           Best value: -7 so it decreases risk of diabetes
#     8. Age (years)
#           Best value: -8.01080076 so it decreases risk of diabetes

# 4) A pure hill-climbing search would not be ideal here. Do you know why?
# The function of decision_entropy is not smooth. That is because you can have multiple weight combinations that lead to a certain amount of women in each group. This leads to the 
# landscape of the function having lots of flat areas where hill-climbing would get stuck.
