from regressionMethods.MultiLinearRegression import  MultiLinearRegression
from regressionMethods.PolynomialRegression import  PoynomialRegression
from regressionMethods.SupportVectorRegression import SupportVectorRegression
from regressionMethods.DecisionTreeRegression  import DecisisionTreeRegression

obj = MultiLinearRegression("Data.csv")
result = obj.train_model()
print(result[0])

obj = PoynomialRegression("Data.csv")
result = obj.train_model()
print(result[0])
obj = SupportVectorRegression("Data.csv")
result = obj.train_model()
print(result[0])

obj = DecisisionTreeRegression("Data.csv")
result = obj.train_model()
print(result[0])