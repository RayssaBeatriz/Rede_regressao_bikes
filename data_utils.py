import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path):
    # Carregamento
    bike = pd.read_csv(csv_path)

    # Limpeza básica e Indexação
    bike = bike.drop(['casual', 'registered', 'instant'], axis=1)
    bike.dteday = pd.to_datetime(bike.dteday, format='%m/%d/%Y')
    bike.index = pd.to_datetime(bike.dteday)
    bike_raw = bike.copy() # Cópia para visualização se necessário
    bike = bike.drop('dteday', axis=1)

    # Separação Numérico/Categórico
    x_numerico = bike[['temp', 'hum', 'windspeed', 'cnt']]
    x_categorico = bike[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]

    # One-Hot Encoding
    onehotenconder = OneHotEncoder()
    x_categorico = onehotenconder.fit_transform(x_categorico).toarray()
    x_categorico = pd.DataFrame(x_categorico)

    # Reorganização
    x_numerico = x_numerico.reset_index()
    x_junto = pd.concat([x_categorico, x_numerico], axis=1)
    x_junto = x_junto.drop(labels=['dteday'], axis=1)

    # Separação X e Y
    x = x_junto.iloc[:, :-1].values
    y = x_junto.iloc[:, -1:].values

    # Normalização do Alvo (Y)
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y)

    # Divisão Treino/Teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    return bike_raw, x_train, x_test, y_train, y_test, scaler