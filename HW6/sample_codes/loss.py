import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loss of 30 epochs
train_loss = np.array([0.7121, 0.4069, 0.2846, 0.1435, 0.0618, 0.0645, 0.0857, 0.0737, 0.1521, 0.1177, 0.0210, 0.2204, 0.0910, 0.0326, 0.0725, 0.0679, 0.0289, 0.0830, 0.0811, 0.0753, 0.0294, 0.0466, 0.0079, 0.0147, 0.0249, 0.0224, 0.0244, 0.0464, 0.0080, 0.0069])
valid_loss = np.array([0.2316, 0.0792, 0.0508, 0.1087, 0.0355, 0.0102, 0.0077, 0.0039, 0.0029, 0.0027, 0.0126, 0.0042, 0.0107, 0.0064, 0.0025, 0.0036, 0.0028, 0.0064, 0.0079, 0.0029, 0.0010, 0.0011, 0.0013, 0.0012, 0.0009, 0.0011, 0.0010, 0.0012, 0.0009, 0.0009])

epoch = range(1, 31)

plt.plot(epoch, train_loss, color = 'b', label = "train")
plt.plot(epoch, valid_loss, color = 'r', label = "validation")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.title("Learning Curve ")
plt.legend(loc='upper right')

plt.show()

