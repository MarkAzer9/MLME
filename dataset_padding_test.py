from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from data_padding import read_dataset3
import matplotlib.pyplot as plt

data, labels=read_dataset3('train',100)
plt.imshow(data[1001],cmap="gray")
#plt.imshow(data[500].reshape(100,100),cmap="gray")
#plt.imshow(data[28].reshape(100,100),cmap="gray")
