import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
# desactivar las advertencias de tipo UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Cargar los datos
df = pd.read_csv('../../data/fat_data//bodyfat.csv')

# Se eliminará la columna Density porque creemos que el obtener esta variable
# para su uso en el proyecto, tiene mucha más dificultad que las demás variables
# con un gran nivel de correlación
df = df.drop(['Density'], axis=1)

# se actualizarán las variables Weight y Height de  libreas a kg y de pulgadas a centimetros

# convertir Height de pulgadas a centímetros
df['Height'] = df['Height'] * 2.54

# convertir Weight de libras a kilogramos
df['Weight'] = df['Weight'] / 2.2046


# Mostrar las primeras filas del conjunto de datos
print(df.head())

# Resumen estadístico de las variables numéricas
print(df.describe())


# seleccionar variables predictoras
X = df[['Chest', 'Abdomen', 'Thigh', 'Hip', 'Weight', 'Neck']]

# seleccionar variable de respuesta
y = df['BodyFat']

# ajustar el modelo de regresión lineal múltiple
model = LinearRegression()
model.fit(X, y)

# dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio y el coeficiente de determinación
print('Error cuadrático medio:', mean_squared_error(y_test, y_pred))
print('Coeficiente de determinación:', r2_score(y_test, y_pred))

# crear un dataframe de Pandas con los mismos nombres de características utilizados para ajustar el modelo
new_data = pd.DataFrame({'Chest': [95], 'Abdomen': [85], 'Thigh': [55], 'Hip': [100], 'Weight': [70], 'Neck': [35]})

# crear la matriz de características a partir del dataframe
X_new = new_data.values

# hacer la predicción con el modelo
y_pred = model.predict(X_new)

# imprimir la predicción
print(y_pred)