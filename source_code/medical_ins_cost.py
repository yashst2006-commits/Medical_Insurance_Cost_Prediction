import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from sklearn.preprocessing import StandardScaler


model_1 = LinearRegression()

model_2 = Ridge(alpha=1.0)

model_3 = Lasso(alpha=0.01)

scaler = StandardScaler()

df = pd.read_csv("Medical_Insurance_Cost_Prediction/Dataset/medical_insurance_cost.csv")
df = pd.get_dummies(
    df,
    columns=["sex" ,  "smoker" , "region"],
    drop_first=True
    )

X = df.drop(["charges"], axis=1)
Y = df["charges"]

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42) 

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model_1.fit(X_train , Y_train)
Y_pred = model_1.predict(X_test)
mae = mean_absolute_error(Y_test , Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test , Y_pred))
r2 = r2_score(Y_test , Y_pred)
print("mae: ", mae)
print("rmse: ", rmse)
print("R2: ", r2)

model_2.fit(X_train_scaled, Y_train)
Y_pred_ridge = model_2.predict(X_test_scaled)
print("Ridge MAE:", mean_absolute_error(Y_test, Y_pred_ridge))
print("Ridge RMSE:", np.sqrt(mean_squared_error(Y_test, Y_pred_ridge)))
print("Ridge R2:", r2_score(Y_test, Y_pred_ridge))

model_3.fit(X_train_scaled, Y_train)
Y_pred_lasso = model_3.predict(X_test_scaled)
print("Lasso MAE:", mean_absolute_error(Y_test, Y_pred_lasso))
print("Lasso RMSE:", np.sqrt(mean_squared_error(Y_test, Y_pred_lasso)))
print("Lasso R2:", r2_score(Y_test, Y_pred_lasso))

plt.figure()
plt.scatter(Y_test, Y_pred_ridge)


plt.plot(
    [Y_test.min(), Y_test.max()],
    [Y_test.min(), Y_test.max()]
)

plt.xlabel("Actual Insurance Cost")
plt.ylabel("Predicted Insurance Cost")
plt.title("Final Model (Ridge): Actual vs Predicted")
plt.show()
plt.savefig("Medical_Insurance_Cost_Prediction/images/Ridge_Reg_Graph.png")