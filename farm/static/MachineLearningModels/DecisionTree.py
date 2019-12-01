from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv(r"../dataset/FinalDataset.csv")
X = data.iloc[:,0:5]
Y = data.iloc[:,5]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100,shuffle = True)
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                               max_depth=3, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)

joblib.dump(clf_entropy, r"../PickleModels/DecisionTree.pkl")



