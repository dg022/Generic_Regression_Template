import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

class SupportVectorRegression:
    def __init__(self, fileName, split=True, splitsize=0.2):
        self.dataset = pd.read_csv(fileName)
        self.X = self.dataset.iloc[:, :-1].values
        self.Y = self.dataset.iloc[:, -1].values
        self.Y  = self.Y.reshape(len(self.Y),1)
        self.split =  split
        self.splitsize = splitsize

    def train_model(self):
        if(self.split):
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.splitsize, random_state=0)
            sc_X = StandardScaler()
            sc_Y = StandardScaler()
            X_train = sc_X.fit_transform(self.X_train)
            Y_train = sc_Y.fit_transform(self.Y_train)
            regressor = SVR(kernel='rbf')
            regressor.fit(X_train, Y_train)
            Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(self.X_test)))
            return [r2_score(self.Y_test, Y_pred), regressor]


