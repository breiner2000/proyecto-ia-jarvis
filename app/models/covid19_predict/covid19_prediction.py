import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from joblib import dump

covid_df = pd.read_csv('../../data/covid_19_data/covid_19_data.csv')

# Borra las columnas innecesarias
covid_df = covid_df.drop(['Province/State','Country/Region','SNo', 'ObservationDate', 'Last Update'], axis=1)

# Pasa los nombres a sting
covid_df.columns = covid_df.columns.astype(str)

# seleccionar variables predictoras
X = covid_df[['Confirmed', 'Recovered']]

# seleccionar variable de respuesta
y = covid_df['Deaths']

# dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ajustar el modelo de regresión lineal múltiple
model = LinearRegression()
model.fit(X_train, y_train)

# Guarda el modelo entrenado en un archivo
dump(model, 'covid19_trained.joblib')

# hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio y el coeficiente de determinación
print('Error cuadrático medio:', mean_squared_error(y_test, y_pred))
print('Coeficiente de determinación:', r2_score(y_test, y_pred))

# crear un dataframe de Pandas con los mismos nombres de características utilizados para ajustar el modelo
country = {
    'Confirmed': [200],
    'Recovered': [50]
}

new_data = pd.DataFrame(country)

# crear la matriz de características a partir del dataframe
X_new = new_data.values

# hacer la predicción con el modelo
y_pred = model.predict(X_new)

# imprimir la predicción
print(y_pred)
