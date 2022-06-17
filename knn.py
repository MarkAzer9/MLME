from sklearn.neighbors import KNeighborsClassifier
from utils.dataset import read_dataset,flatten
import argparse
import numpy as np

print("loading  training data.........")
train_data,train_labels=read_dataset('train',100)
train_data=flatten(train_data)
train_labels=np.argmax(train_labels,1)

print("inititalizing Knn model.........")

model = KNeighborsClassifier(n_neighbors=10,weights='uniform',n_jobs=None)

print("fitting our training data.........")
model.fit(train_data,train_labels)

print("loading  test data.........")
test_data,test_labels=read_dataset('test',100)
test_data=flatten(test_data)
test_labels=np.argmax(test_labels,1)

print("predicting test data.........")
acc=model.score(test_data,test_labels)
print("[INFO]  accuracy: {:.2f}%".format(acc * 100))