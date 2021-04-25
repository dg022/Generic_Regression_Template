import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class MultiLinearRegression:
    def __init__(self, fileName, split=True, splitsize=0.2):
        self.dataset = pd.read_csv(fileName)
        self.X = self.dataset.iloc[:, :-1].values
        self.Y = self.dataset.iloc[:, -1].values
        self.split =  split
        self.splitsize = splitsize

    def train_model(self):
        if(self.split):
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.splitsize, random_state=0)
            regressor = LinearRegression()
            regressor.fit(self.X_train, self.Y_train)
            self.Y_pred = regressor.predict(self.X_test)
            return [r2_score(self.Y_test, self.Y_pred), regressor]
        else:
            regressor = LinearRegression()
            regressor.fit(self.X, self.Y)
            self.Y_pred = regressor.predict(self.X)
            return [r2_score(self.Y, self.Y_pred), regressor]

