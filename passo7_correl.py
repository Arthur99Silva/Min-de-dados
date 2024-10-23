import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Análise de Correlações

# Carregar o conjunto de dados
spotify_data = pd.read_csv("spotify-2023.csv", encoding='ISO-8859-1')

# Filtrar apenas as colunas numéricas para calcular a matriz de correlação
numeric_cols = spotify_data.select_dtypes(include=['float64', 'int64'])

# Criando um heatmap de correlação para variáveis numéricas
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.show()
