import pandas as pd
import numpy as np
from joblib import load
import os

# obtener la ruta completa del archivo del modelo entrenado
wine_model_path = os.path.join(os.path.dirname(__file__), 'wine_quality', 'wine_quality_trained.joblib')


def classify_wine_quality(wine_dict):
    # load the trained model
    rfc = load(wine_model_path)

    # make a df with the new wine
    wine_df = pd.DataFrame([wine_dict])
    wine_df['free sulfur dioxide'] = np.log(1 + wine_df['free sulfur dioxide'])

    # make the prediction
    wine_quality = rfc.predict(wine_df)
    print('La calidad estimada del vino es:', wine_quality)
    return wine_quality
