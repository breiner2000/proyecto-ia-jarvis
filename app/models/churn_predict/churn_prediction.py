import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
data = pd.read_csv(url)

# Visualizar los primeros registros
data.head()

# Eliminar la columna "customerID" ya que no aporta información relevante
data = data.drop('customerID', axis=1)

# Reemplazar los valores "No internet service" y "No phone service" por "No"
data = data.replace({'No internet service': 'No', 'No phone service': 'No'})

# Convertir la columna "TotalCharges" en numérica
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Imputar los valores faltantes con la media

imputer = SimpleImputer(strategy='mean')

data['TotalCharges'] = imputer.fit_transform(data[['TotalCharges']])


le = LabelEncoder()
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'Churn']


for col in cat_cols:
    data[col] = le.fit_transform(data[col])


# Separar las variables de entrada y salida
X = data.drop('Churn', axis=1)
y = data['Churn']


# Dividir los datos en conjuntos de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Crear el modelo de Regresión Logística
model = LogisticRegression(max_iter=1000)

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Predecir los valores de salida con los datos de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo

accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo: {:.2f}%".format(accuracy * 100))



# 'gender': ['Female', 'Male']
# 'Partner': ['No', 'Yes']
# 'Dependents': ['No', 'Yes']
# 'PhoneService': ['No', 'Yes']
# 'MultipleLines': ['No', 'Yes']
# 'InternetService': ['DSL', 'Fiber optic', 'No']
# 'OnlineSecurity': ['No', 'Yes']
# 'OnlineBackup': ['No', 'Yes']
# 'DeviceProtection': ['No', 'Yes']
# 'TechSupport': ['No', 'Yes']
# 'StreamingTV': ['No', 'Yes']
# 'StreamingMovies': ['No', 'Yes']
# 'Contract': ['Month-to-month', 'One year', 'Two year']
# 'PaperlessBilling': ['No', 'Yes']
# 'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)','Electronic check', 'Mailed check']
# 'Churn': ['No', 'Yes']



new_data = pd.DataFrame({
    "gender" :[0],
    "SeniorCitizen":[0],
    "Partner":[0],
    "Dependents":[0],
    "tenure":[2],
    "PhoneService":[1],
    "MultipleLines":[0],
    "InternetService":[1],
    "OnlineSecurity":[0],
    "OnlineBackup":[0],
    "DeviceProtection":[0],
    "TechSupport":[0],
    "StreamingTV":[0],
    "StreamingMovies":[0],
    "Contract":[0],
    "PaperlessBilling":[1],
    "PaymentMethod":[2],
    "MonthlyCharges":[70],
    "TotalCharges":[151]
})


# Hacer la predicción
prediction = model.predict(new_data)

if (prediction == 0):
    print("La persona no es propensa a cambiar de compañía")
else:
    print("La persona sí es propensa a cambiar de compañía")






