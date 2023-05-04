import pandas as pd

from joblib import dump, load  # import dump functions and load for save and load the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split


# Cargar los datos
data = pd.read_csv('../../data/walmart_data/walmartsales.csv')

# Crear características adicionales
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Semester'] = data['Month'].apply(lambda x: 1 if x <= 6 else 2)

data['IsHoliday'] = data['IsHoliday'].apply(lambda x: 0 if x == False else 1)

# Seleccionar características relevantes
features = ['Store', 'Dept', 'Month', 'Semester', 'IsHoliday']
X = data[features]
y = data['Weekly_Sales']


# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de Bosque Aleatorio y ajustar los hiperparámetros
rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE en el conjunto de prueba: ", mae)

X_new = pd.DataFrame({'Store': [10], 'Dept': [1], 'Month': [8], 'Semester': [2], 'IsHoliday': [0]})

prediction = rf.predict(X_new)

print(prediction)

# save the trained model into a file
dump(rf, 'walmart_sales_trained.joblib')