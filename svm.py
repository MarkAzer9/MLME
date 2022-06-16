
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from utils.dataset import read_dataset,flatten

svm = SVC(kernel='rbf',C=100,gamma=0.01,degree=9,verbose=True)

train_data,train_labels=read_dataset('train',100)
train_data=flatten(train_data)
train_labels=np.argmax(train_labels,1)

svm.fit(train_data, train_labels)

test_data,test_labels=read_dataset('test',100)
test_data=flatten(test_data)
test_labels=np.argmax(test_labels,1)

pred = svm.predict(test_data)
print(accuracy_score(test_labels, pred)) # Accuracy
confusion=confusion_matrix(test_labels,pred)

