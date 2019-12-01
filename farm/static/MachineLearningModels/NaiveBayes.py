import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


data = pd.read_csv(r"../dataset/FinalDataset.csv")
X = data.iloc[:,0:5]
Y = data.iloc[:,5]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100,shuffle = True)

gaussian_clf = GaussianNB()
gaussian_clf.fit(X_train,y_train)

joblib.dump(gaussian_clf, r"../PickleModels/NB.pkl")




