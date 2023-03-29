import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from joblib import dump, load  # import dump functions and load for save and load the model

# %matplotlib inline
warnings.filterwarnings('ignore')

df = pd.read_csv('../../data/wine_data/winequalityN.csv')

# filling of the missing values
for col, value in df.items():
    if col != 'type':  # ignoring the 'type' column because the value is in string
        df[col] = df[col].fillna(df[col].mean())

# log transformation, because 'free sulfur dioxide' was in big range
df['free sulfur dioxide'] = np.log(1 + df['free sulfur dioxide'])

# dropping the 'type' and 'quality' from the dataset
X = df.drop(columns=['type', 'quality', 'sulphates', 'pH', 'total sulfur dioxide', 'residual sugar', 'citric acid',
                     'fixed acidity'])
y = df['quality']

# using 'SMOTE', because it gets average features from the neighbours, and create new features
oversample = SMOTE(k_neighbors=4)
# transform the dataset
X, y = oversample.fit_resample(X, y)

# modelo random forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

rfc = RandomForestClassifier()
# Fit the model
rfc.fit(X_train, y_train)

# save the trained model into a file
dump(rfc, 'wine_quality_trained.joblib')
