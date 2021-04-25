from regressionMethods.MultiLinearRegression import  MultiLinearRegression


obj = MultiLinearRegression("Data.csv")
result = obj.train_model()
print(result[0])
