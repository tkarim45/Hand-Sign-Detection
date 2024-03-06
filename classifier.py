import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# making data dict to store data and labels
data_dict = pickle.load(open('data.pickle', 'rb'))

# converting data and labels to numpy arrays
data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

# Splitting the data into test and train. Shuffle the data according to labels.
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Apply svm model to train the model to predict the hand signs
model = svm.SVC(kernel='linear', C=1, gamma='auto')
model.fit(X_train, y_train)

# we apply the model of text data to check the performance of the model
y_pred = model.predict(X_test)

# printing the accuracy of the model on the test data
print(accuracy_score(y_test, y_pred))

# we are storing the model in model.p file later to check the prediction of hand signs live
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
