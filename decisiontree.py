import pandas as pd
import pydotplus
from IPython.display import Image
from six import StringIO
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

df = pd.read_csv("csv/vehicle.csv")
X = df.iloc[:,:18]
y = df.iloc[:,18]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(criterion="entropy")

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['Van','Saab','Opel','Bus'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('plots/decisiontree.png')
Image(graph.create_png())