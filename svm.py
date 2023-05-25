import pandas as pd
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

df = pd.read_csv("csv/vehicle.csv")
X = df.iloc[:,:18]
y = df.iloc[:,18]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
