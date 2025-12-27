import matplotlib.pyplot as plt
import seaborn as sns

def visualizar_dados_iniciais(bike, x_numerico):
    # Pre-processamento e visualização dos dados (Heatmap de nulos)
    sns.heatmap(bike.isnull()) 
    plt.show()
    plt.close()

    # Visualização dos dados temporais
    bike['cnt'].asfreq('W').plot()
    plt.title('Bikes alugadas por semana')
    plt.xlabel('Semana')
    plt.ylabel('Bikes alugadas')
    plt.show()
    plt.close()

    bike['cnt'].asfreq('ME').plot()
    plt.title('Bikes alugadas por mês')
    plt.xlabel('Mês')
    plt.ylabel('Bikes alugadas')
    plt.show()
    plt.close()

    sns.pairplot(bike)
    plt.show()
    plt.close()

    sns.pairplot(x_numerico)
    plt.show()
    plt.close()

    # Correlação entre as variáveis numéricas
    sns.heatmap(x_numerico.corr(), annot=True)
    plt.show()
    plt.close()

def visualizar_historico_treino(epochs_hist):
    plt.plot(epochs_hist.history['loss'], label='loss')
    plt.plot(epochs_hist.history['val_loss'], label='val_loss')
    plt.xlabel('Épocas')
    plt.ylabel('Erro de treinamento e validação')
    plt.legend(['Erro de treinamento', 'Erro de validação'])
    plt.show()

def visualizar_predicoes(y_test, y_predict):
    plt.plot(y_test, y_predict, "^")
    plt.xlabel('Valores reais')
    plt.ylabel('Predição do modelo')
    plt.show()
    plt.close()

