# 🚲 Previsão de Aluguel de Bicicletas

Projeto de Redes Neurais com **TensorFlow** para prever a demanda diária de aluguel de bicicletas. O código foi modularizado para facilitar a manutenção e escalabilidade.

## 📂 Estrutura

* `main.py`: Script principal para execução.
* `data_utils.py`: Carregamento, limpeza e pré-processamento (One-Hot Encoding, Normalização).
* `neural_net.py`: Arquitetura da Rede Neural (Camadas Densas).
* `graphics.py`: Geração de gráficos para análise e métricas.

## 🛠 Tecnologias

`Python` | `TensorFlow/Keras` | `Pandas` | `Scikit-Learn` | `Seaborn`

## 🚀 Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/RayssaBeatriz/Rede_regressao_bikes
    cd Rede_regressao_bikes
    ```

2.  **Instale as dependências:**
    ```bash
    pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Rode o projeto:**
    ```bash
    python main.py
    ```
Obs: caso deseje exibir os gráficos de visualização dos dados, mude a váriavel SHOW_EDA_GRAPHS para True no arquivo main.py

## 📊 Modelo e Resultados

O modelo utiliza uma Rede Neural Artificial (3 camadas ocultas de 100 neurônios, ativação ReLU) e avalia o desempenho utilizando métricas como **MSE**, **RMSE** e **R²**.
