from regressionMethods.MultiLinearRegression import  MultiLinearRegression
from regressionMethods.PolynomialRegression import  PoynomialRegression


obj = MultiLinearRegression("Data.csv")
result = obj.train_model()
print(result[0])

obj = PoynomialRegression("Data.csv")
result = obj.train_model()
print(result[0])