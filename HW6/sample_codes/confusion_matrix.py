import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix

true = pd.read_csv("../digit/valid.csv")
test = pd.read_csv("./test_valid.csv")

true_label = true["label"].to_numpy()
test_label = test["label"].to_numpy()
label_num = np.arange(10)

cm = confusion_matrix(true_label, test_label)

sns.heatmap(cm, cmap = "Reds", annot = True, cbar = True, fmt='d', xticklabels = label_num, yticklabels = label_num)
plt.xlabel("Predict")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.show()

