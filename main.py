import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# Importando nossos módulos
import data_utils
import graphics
import neural_net

# Configurações
CSV_FILE = 'bike-sharing-daily.csv'
SHOW_EDA_GRAPHS = False  # Mude para True se quiser ver os gráficos iniciais

def main():
    print("--- 1. Carregando e Processando Dados ---")
    bike_raw, x_train, x_test, y_train, y_test, scaler = data_utils.load_and_preprocess_data(CSV_FILE)
    
    if SHOW_EDA_GRAPHS:
        print("--- Exibindo Gráficos de Análise Exploratória ---")
        graphics.plot_eda(bike_raw)

    print("--- 2. Construindo e Treinando o Modelo ---")
    input_shape = x_train.shape[1]
    modelo = neural_net.build_model(input_shape)
    print(modelo.summary())

    epochs_hist = modelo.fit(
        x_train, 
        y_train, 
        epochs=9, 
        batch_size=40, 
        validation_split=0.2201,
        verbose=1
    )

    # Plotar histórico de treinamento
    graphics.visualizar_historico_treino(epochs_hist)

    print("--- 3. Avaliação do Modelo ---")
    y_predict = scaler.inverse_transform(modelo.predict(x_test))
    y_test_inverse = scaler.inverse_transform(y_test)
    
    # Plotar predições
    graphics.visualizar_predicoes(y_test_inverse, y_predict)

    # Métricas
    k = x_test.shape[1]
    n = len(x_test)
    
    mae = mean_absolute_error(y_test_inverse, y_predict)
    mse = mean_squared_error(y_test_inverse, y_predict)
    rmse = sqrt(mse)
    r2 = r2_score(y_test_inverse, y_predict)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    print(f"\nResultados Finais:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"ADJ R2: {adj_r2:.4f}")

if __name__ == "__main__":
    main()