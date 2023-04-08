import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from joblib import dump, load

# Cargamos los datos en un DataFrame de Pandas
df = pd.read_csv('../../data/avocado_data/avocado.csv')


# Mostrar las primeras filas del conjunto de datos
print(df.head())

# Resumen estadístico de las variables numéricas
print(df.describe())

# Eliminamos la columna 'Unnamed: 0'
df.drop('Unnamed: 0', axis=1, inplace=True)


# Seleccionamos las variables predictoras
X = df[['Total Volume', '4046', '4225', '4770', 'type', 'year', 'region']]

# Seleccionamos la variable objetivo
y = df['AveragePrice']


# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocesamiento de datos
num_transformer = StandardScaler()

cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', num_transformer, ['Total Volume', '4046', '4225', '4770', 'year']),
                                               ('cat', cat_transformer, ['type', 'region'])])

# Creamos un modelo de regresión RandomForestRegressor
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Entrenamos el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

dump(model, 'avocado_price_trained.joblib')

# Evaluamos el modelo con los datos de prueba
y_pred = model.predict(X_test)
print('Error cuadrático medio:', mean_squared_error(y_test, y_pred))
score = r2_score(y_test, y_pred)
print("R2 score:", score)

new_data = pd.DataFrame({'Total Volume': [100], '4046': [20], '4225': [30], '4770': [10],
                         'type': ['conventional'], 'year': [2016], 'region': ['Albany']})

prediction = model.predict(new_data)
print('El precio de venta estimado es de:',prediction)

