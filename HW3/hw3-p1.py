import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import time

# A vector containing the names of the 2000 genes for the gene expression matrix x.
indexes = pd.read_csv('hw3_Data1/index.txt', delimiter = '\t', header = None)
# print(indexes)

# A (62 x 2000) matrix giving the expression levels of 2000 genes for the 62 Colon tissue samples. 
# Each row corresponds to a patient, each column to a gene.
x_ = pd.read_csv('hw3_Data1/gene.txt', delimiter = ' ', header = None)#.to_numpy().T
x = x_.to_numpy().T
# print(x_)

# A numeric vector of length 62 giving the type of tissue sample (tumor or normal).
y = pd.read_csv('hw3_Data1/label.txt', header = None).to_numpy()
y = (y > 0).astype(int).reshape(y.shape[0])
# print(sum(y == 1)) #22
# print(sum(y == 0)) #40
# print(type(y))

# Only take the name of genes
indexes_name = indexes.iloc[:, 0]
pd.options.mode.chained_assignment = None 
for i in range(2000):
	indexes_name[i] = indexes_name[i].strip()
	# print(indexes_name[i])

# Make the table of 2000 features with 62 samples
x_.index = indexes_name
x_ = x_.T
# print(x_)
# x_.to_csv("output.csv", index = False)

# Calculate mean of every expression levels and rank it
idx_mean = x_.mean().to_frame()
# print(type(idx_mean))
# print(idx_mean)
idx_mean_rank = idx_mean.rank()
idx_mean_rank = idx_mean_rank.reset_index(drop = True)
idx_mean_rank = idx_mean_rank.astype(int)
# print(idx_mean_rank) 
ranking_idx = idx_mean_rank[0].to_numpy() - 1
# print(ranking_idx)
# print(len(ranking_idx))

# Feature evaluation 
# Use a simple random forest with 5-fold validation to evaluate the feature selection result.
score_history_RF = []
score_history_DT = []
for m in range(5, 2001, 5):
    # Select Top m feature
    x_subset = x[:, ranking_idx[:m]]

    # Build random forest and decision tree
    clf_RF = RandomForestClassifier(random_state = 0)
    clf_DT = DecisionTreeClassifier(random_state = 0)

    # Calculate validation score
    scores_RF = cross_val_score(clf_RF, x_subset, y, cv = 5)
    scores_DT = cross_val_score(clf_DT, x_subset, y, cv = 5)

    # Save the score calculated with m feature
    score_history_RF.append(scores_RF.mean())
    score_history_DT.append(scores_DT.mean())

    # Show progress bar
    print(f'目前進度：{m * 100 / 2000} %\r', end='')
    time.sleep(0.1)

# Report best accuracy.
print(f"Max of Decision Tree: {max(score_history_DT)}")
print(f"Number of features: {np.argmax(score_history_DT) * 5 + 5}")
print(' ')
print(f"Max of Random Forest: {max(score_history_RF)}")
print(f"Number of features: {np.argmax(score_history_RF) * 5 + 5}")

# Visualization
plt.plot(range(5, 2001, 5), score_history_RF, c = 'blue')
plt.title('Original')
plt.xlabel('Number of features')
plt.ylabel('Cross-validation score')
plt.legend(['Random Forest'])
plt.savefig('hw3_p1_RF_result.png')

# Get the subsets
pos = np.zeros(2000).astype(int)
for i in range(2000):
    if ranking_idx[i] < (np.argmax(score_history_RF) * 5 + 5):
        pos[i] = 1
select = np.where(pos == 1)
select = np.asarray(select).flatten()
opt_select = indexes.iloc[select]
pd.options.mode.chained_assignment = None

# Save optimizer subsets
for i in range(opt_select.shape[0]):
    opt_select.iloc[i, 0] = opt_select.iloc[i, 0].strip()
# print(opt_select)
opt_select.to_csv("p1_opt_select.csv", index = False)