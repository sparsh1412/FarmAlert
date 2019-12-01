import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r"../dataset/FinalDataset.csv")
dataset_x = dataset.iloc[:, :-1]
dataset_y = dataset.iloc[:, -1]

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)

x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=100, shuffle=True)

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_train)


joblib.dump(knn_clf, r"../PickleModels/KNN.pkl")
