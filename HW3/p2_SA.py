import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SA import train_model, simulated_annealing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

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


# SA
y_train = pd.DataFrame(y, columns = ["Survived"])
# print(y_train)

results, best_metric, best_subset_cols, best_subset = simulated_annealing(x_, y_train)

x_subset = x[:, list(best_subset)]

# Build random forest
clf = RandomForestClassifier(random_state=0)

# Calculate validation score
scores = cross_val_score(clf, x_subset, y, cv=5).mean()

print("\n\n")
print(f"Max of Decision Tree: {scores}")
print(f"Number of features: {len(best_subset_cols)}")
print("The best metrix is {0}".format(best_metric))