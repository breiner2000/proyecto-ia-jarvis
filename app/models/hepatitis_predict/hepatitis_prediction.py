import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from joblib import dump

# Cargar los datos
df = pd.read_csv('../../data/hepatitis_data/HepatitisCdata.csv')

# Pasa los categories a int que equivalen a un ENUM
category_map = {'0=Blood Donor': 0, '1=Hepatitis': 1, '2=Fibrosis': 2,
                '3=Cirrhosis': 3}

# Se eliminará la columna Category
df['Category'] = df['Category'].map(category_map)

# Mostrar las primeras filas del conjunto de datos
print(df.head())

# Resumen estadístico de las variables numéricas
print(df.describe())

# seleccionar variables predictoras
X = df[['Age', 'ALT', 'CHOL', 'PROT', 'ALB', 'AST']]

# seleccionar variable de respuesta
y = df['Category']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = y.fillna(y.mean())

print(pd.DataFrame(X).isnull().sum())

# dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

# save the trained model into a file
dump(model, 'hepatitis_trained.joblib')

# hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio y el coeficiente de determinación
print('Error cuadrático medio:', mean_squared_error(y_test, y_pred))
print('Coeficiente de determinación:', r2_score(y_test, y_pred))

# crear un dataframe de Pandas con los mismos nombres de características utilizados para ajustar el modelo
person = {
            'Age': [32],
            'ALT': [7.7],
            'CHOL': [3.23],
            'PROT': [69],
            'ALB': [38.5],
            'AST': [22.1]
}


new_data = pd.DataFrame(person)

# crear la matriz de características a partir del dataframe
X_new = new_data.values

# hacer la predicción con el modelo
y_pred = model.predict(X_new)

# imprimir la predicción
print(y_pred)




