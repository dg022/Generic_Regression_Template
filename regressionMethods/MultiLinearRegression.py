import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MultiLinearRegression:
    def __init__(self, fileName):
        self.dataset = pd.read_csv(fileName)
        self.X = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, -1].values