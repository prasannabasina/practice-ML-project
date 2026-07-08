import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df=pd.read_csv("house_dataset.csv")
print(df.head())
x=df[["area","bedroom","bathroom"]]
y=df["price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
new_house=pd.DataFrame({
  "area":[2400],
  "bedroom":[4],
  "bathroom":[3]
})
price=model.predict(new_house)
print(price)
mse=mean_squared_error(y_test,prediction)
print("mean squared error",mse)

