from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.Data_Preprocessing2 import read_dataset, padd_images, flatten
import numpy as np



# param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100],'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],'kernel': ['rbf','poly', 'linear']}
# svm = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5,)
#svm = SVC(kernel='poly', C=0.5, gamma=0.001, degree=9, verbose=True)
#svm = SVC(kernel='rbf', C=10, gamma=0.01, degree=9, verbose=True)
svm = SVC(kernel='rbf', C=100, gamma=0.01, degree=9, verbose=True)
train_data, train_labels, dim_train = read_dataset('train')



train_data = padd_images(train_data,dim_train)
train_data = flatten(train_data)
train_labels = np.argmax(train_labels, 1)

n = 66
scaler = StandardScaler()
pca = PCA(n_components=n)
#scaler.fit(train_data)


#pca.fit(train_data)# Normalize the data first for  Principal component analysis

train_data = pca.fit_transform(train_data)
#train_data = scaler.fit_transform(train_data)
svm.fit(train_data, train_labels)

# best_params = svm.best_params_
# print(f"Best params: {best_params}")



test_data, test_labels, dim_test = read_dataset('test')
test_data = padd_images(test_data, dim_train)
test_data = flatten(test_data)
test_labels = np.argmax(test_labels, 1)


test_data = pca.transform(test_data)
#test_data = scaler.transform(test_data)  # Normalize the data first for  Principal component analysis


# svm_clf = SVC(**best_params)
# svm_clf.fit(train_data, train_labels)
# print_score(svm_clf, train_data, train_labels, test_data, test_labels, train=True)
# print_score(svm_clf, train_data, train_labels, test_data, test_labels, train=False)
pred = svm.predict(test_data)
print(accuracy_score(test_labels, pred))  # Accuracy
# confusion = confusion_matrix(test_labels, pred)

print("Hello World")
