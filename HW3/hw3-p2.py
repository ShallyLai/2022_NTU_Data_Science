import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyswarms as ps
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# A vector containing the names of the 2000 genes for the gene expression matrix x.
indexes = pd.read_csv('hw3_Data1/index.txt', delimiter = '\t', header = None)

# A (62 x 2000) matrix giving the expression levels of 2000 genes for the 62 Colon tissue samples. 
# Each row corresponds to a patient, each column to a gene.
X = pd.read_csv('hw3_Data1/gene.txt', delimiter = ' ', header = None).to_numpy().T

# A numeric vector of length 62 giving the type of tissue sample (tumor or normal).
y = pd.read_csv('hw3_Data1/label.txt', header = None).to_numpy()
y = (y > 0).astype(int).reshape(y.shape[0])

# Create an instance of the classifier
classifier = RandomForestClassifier(random_state = 87)

# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = X.shape[1]
    
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:, m == 1]
   
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    
    # Compute for the objective function
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha = 0.87):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    X: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm, arbitrary
options = {'c1': 0.6, 'c2': 0.9, 'w': 0.3, 'k': 20, 'p': 2}

# Call instance of PSO
dimensions = X.shape[1] # dimensions should be the number of features
optimizer = ps.discrete.BinaryPSO(n_particles = 30, dimensions = dimensions, options = options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters = 30)
print('\nSelected features : ' + str(sum((pos == 1) * 1)) + ' / ' + str(len(pos)))

# Get best subset position
select = np.where(pos == 1)
select = np.asarray(select).flatten()
# print(select)
# print(type(select))

# Build random forest
classifier = RandomForestClassifier(random_state = 87)

# Get the selected features from the final positions
X_selected_features = X[:, pos == 1]  # subset

# Perform classification and store performance in P
subset_performance = cross_val_score(classifier, X_selected_features, y, cv = 5).mean()

# Compute performance
print('Subset performance: %.3f' % (subset_performance))

# Selected subsets
opt_select = indexes.iloc[select]
pd.options.mode.chained_assignment = None
for i in range(opt_select.shape[0]):
    opt_select.iloc[i, 0] = opt_select.iloc[i, 0].strip()
    # print(opt_select.iloc[i, 0])
opt_select.to_csv("p2_opt_select.csv", index = False)
