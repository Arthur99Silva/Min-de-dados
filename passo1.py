import pandas as pd

# Carregar o conjunto de dados fornecido
spotify_data = pd.read_csv("spotify-2023.csv", encoding='ISO-8859-1')

# Exibir as primeiras linhas para garantir que foi carregado corretamente
print(spotify_data.head())
