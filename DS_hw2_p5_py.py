import numpy as np
import pandas as pd
 
class1 = np.array([[5, 3], [3, 5], [3, 4], [4, 5], [4, 7], [5, 6]]) 
class2 = np.array([[9, 10], [7, 7], [8, 5], [8, 8], [7, 2], [10, 8]])

mean1 = np.array(np.mean(class1, 0))
mean2 = np.array(np.mean(class2, 0))
total_mean = np.array((mean1 + mean2) /2)
# print("Mean1:\n{0}\nmean2:\n{1}".format(pd.DataFrame(mean1), pd.DataFrame(mean2)))
# print(pd.DataFrame(total_mean))

Sb1 = np.matmul((mean1 - total_mean).reshape(2, 1), (mean1 - total_mean).reshape(2, 1).T)
Sb2 = np.matmul((mean2 - total_mean).reshape(2, 1), (mean2 - total_mean).reshape(2, 1).T) 
SB = Sb1 + Sb2
# print(SB)

Sw1 = np.matmul((class1 - mean1).T, (class1 - mean1))
Sw2 = np.matmul((class2 - mean2).T, (class2 - mean2))
SW = Sw1 + Sw2
# print(SW)

A = np.matmul(np.linalg.inv(SW), SB) 
eigvalue, eigvector = np.linalg.eig(A)
print("Eigenvector:\n{0}\n".format(eigvector))
print("Eigenvalue:\n{0}\n".format(eigvalue))
