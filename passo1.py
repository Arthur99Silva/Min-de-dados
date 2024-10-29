import pandas as pd

# Carregar conjunto
spotify_data = pd.read_csv("spotify-2023.csv", encoding='ISO-8859-1')

print(spotify_data.head())
