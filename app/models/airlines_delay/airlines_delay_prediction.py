import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from joblib import dump

# Cargar los datos
df = pd.read_csv('../../data/airlines_data/DelayedFlights.csv')

# Mostrar las primeras filas del conjunto de datos
print(df.head())

# Resumen estadístico de las variables numéricas
print(df.describe())

# seleccionar variables predictoras
X = df[['Distance', 'DayOfWeek', 'DepTime', 'TaxiOut', 'TaxiIn', 'ArrDelay']]

# seleccionar variable de respuesta
y = df['DepDelay']

# Imputar los valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = y.fillna(y.mean())

print(pd.DataFrame(X).isnull().sum())

# dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ajustar el modelo de regresión lineal múltiple
model = LinearRegression()
model.fit(X_train, y_train)

# Guarda el modelo entrenado en un archivo
dump(model, 'airlines_trained.joblib')

# hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio y el coeficiente de determinación
print('Error cuadrático medio:', mean_squared_error(y_test, y_pred))
print('Coeficiente de determinación:', r2_score(y_test, y_pred))

# crear un dataframe de Pandas con los mismos nombres de características utilizados para ajustar el modelo
airlane = {
            'Distance': [810],
            'DayOfWeek': [4],
            'DepTime': [628.0],
            'TaxiOut': [4.0],
            'TaxiIn': [8.0],
            'ArrDelay': [-14.0]
}


new_data = pd.DataFrame(airlane)

# crear la matriz de características a partir del dataframe
X_new = new_data.values

# hacer la predicción con el modelo
y_pred = model.predict(X_new)

# imprimir la predicción
print(y_pred)


