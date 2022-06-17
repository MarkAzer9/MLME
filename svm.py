
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from utils.dataset import read_dataset,flatten,Principle_Components
import numpy as np

#param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100],'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],'kernel': ['rbf', 'poly', 'linear']}
#svm = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
svm = SVC(kernel='rbf',C=100,gamma=0.01,degree=9,verbose=True)

train_data,train_labels=read_dataset('train',100)
train_data=flatten(train_data)
train_labels=np.argmax(train_labels,1)
n_components=50

train_data=Principle_Components(train_data,n_components)

svm.fit(train_data, train_labels)

#best_params = svm.best_params_
#print(f"Best params: {best_params}")


test_data,test_labels=read_dataset('test',100)
test_data=flatten(test_data)

test_data=Principle_Components(test_data,n_components)
test_labels=np.argmax(test_labels,1)


pred = svm.predict(test_data)
print(accuracy_score(test_labels, pred)) # Accuracy
confusion=confusion_matrix(test_labels,pred)

