import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from joblib import dump, load  # import dump functions and load for save and load the model

car_df = pd.read_csv('../../data/car_data/car_data.csv')
data = pd.get_dummies(car_df, drop_first=True)

# Encoding "Fuel_Type Column"
car_df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# Encoding "Seller Column"
car_df.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# Encoding "Transmission Column"
car_df.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

# dividir data en prueba y entrenamiento
X=car_df.drop(['Car_Name','Selling_Price'],axis=1)
y=car_df['Selling_Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=2)

lin_reg=LinearRegression()

lin_reg.fit(X_train,y_train)

# Prediction on testing data
Test_data_pred=lin_reg.predict(X_test)

error_score=metrics.r2_score(y_test,Test_data_pred)
print("R squared error", error_score)

# save the trained model into a file
dump(lin_reg, 'car_price_trained.joblib')

# vars: Year	Present_Price	Kms_Driven	Fuel_Type	Seller_Type	Transmission	Owner
# Fuel_Type: 0-Petrol, 1-Diesel, 2-CNG
# Seller_Type: 0-Dealer, 1-Individual
# Transmission: 0-Manual, 1-Automatic

# predecir precio de un vehiculo
car = {
        'Year': 2016,
        'Present_Price':8.01,
        'Kms_Driven': 10000,
        'Fuel_Type': 1,
        'Seller_Type': 0,
        'Transmission': 0,
        'Owner': 0
        }

car_df = pd.DataFrame([car])

# clasificar el vino
selling_price = lin_reg.predict(car_df)
print('El precio de venta estimado es de:', selling_price)
