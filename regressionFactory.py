from regressionMethods.MultiLinearRegression import  MultiLinearRegression
from regressionMethods.PolynomialRegression import  PoynomialRegression
from regressionMethods.SupportVectorRegression import SupportVectorRegression
from regressionMethods.DecisionTreeRegression  import DecisisionTreeRegression
from regressionMethods.RandomForestRegression import RandomForestRegression
obj = MultiLinearRegression("CroqPain.csv")
result = obj.train_model()
print(result[0])

obj = PoynomialRegression("CroqPain.csv")
result = obj.train_model()
print(result[0])
obj = SupportVectorRegression("CroqPain.csv")
result = obj.train_model()
print(result[0])

obj = DecisisionTreeRegression("CroqPain.csv")
result = obj.train_model()
print(result[0])

obj = RandomForestRegression("CroqPain.csv")
result = obj.train_model()
print(result[0])