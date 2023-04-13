import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

data = pd.read_csv(r'C:\Users\User\Desktop\Osnove strojnog ucenja\LV4\data_C02_emission.csv')

# a)
variables = data[["Engine Size (L)", 
          "Fuel Consumption City (L/100km)", 
          "Fuel Consumption Hwy (L/100km)", 
          "Cylinders",
          "Fuel Consumption Comb (L/100km)", 
          "Fuel Consumption Comb (mpg)"]]

output = ['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(variables, output, test_size = 0.2, random_state = 1)

# b)

plt.scatter(X_train["Fuel Consumption City (L/100km)"], y_train, color="blue", s=2, label="Train")
plt.scatter(X_test["Fuel Consumption City (L/100km)"], y_test, color="red", s=2, label="Test")
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")

plt.legend()
plt.show()

# c)

sc = MinMaxScaler()
X_train_n = sc.fit_transform( X_train )
X_test_n = sc.transform( X_test )


plt.hist( X_train[:, 3], bins = 10)
plt.figure()
plt.hist(X_train_n[:, 3], bins = 10)
plt.show()

# d)

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)

# e)

y_test_p = linearModel.predict(X_test_n )

plt.scatter(X_test["Fuel Consumption City (L/100km)"], y_test_p, color="blue", label="Predicted data", s=1)
plt.scatter(X_test["Fuel Consumption City (L/100km)"], y_test, color="red", label="True data", s=1)
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
plt.show()

# f) g)
MSE = mean_squared_error(y_test, y_test_p)
MAE = mean_absolute_error(y_test , y_test_p)
r2 = r2_score(y_test, y_test_p)
