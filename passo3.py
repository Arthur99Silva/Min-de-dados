import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

spotify_data = pd.read_csv("spotify-2023.csv", encoding='ISO-8859-1')

# Convertendo a coluna 'streams' para numérico
spotify_data['streams'] = pd.to_numeric(spotify_data['streams'].str.replace(',', ''), errors='coerce')

print("Estatísticas Descritivas das Variáveis Numéricas:")
print(spotify_data.describe())

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Distribuição de danceability_%
sns.histplot(spotify_data['danceability_%'], bins=20, kde=True, ax=axs[0])
axs[0].set_title('Distribuição de Danceability (%)')
axs[0].set_xlabel('Danceability (%)')
axs[0].set_ylabel('Frequência')

# Distribuição de energy_%
sns.histplot(spotify_data['energy_%'], bins=20, kde=True, ax=axs[1])
axs[1].set_title('Distribuição de Energy (%)')
axs[1].set_xlabel('Energy (%)')
axs[1].set_ylabel('Frequência')

# Distribuição de streams
sns.histplot(spotify_data['streams'].dropna(), bins=20, kde=True, ax=axs[2])
axs[2].set_title('Distribuição de Streams')
axs[2].set_xlabel('Streams')
axs[2].set_ylabel('Frequência')

plt.tight_layout()
plt.show()


# Distribuição de 'mode' (Major/Minor)
plt.figure(figsize=(8, 6))
sns.countplot(data=spotify_data, x='mode')
plt.title('Distribuição dos Modos (Major/Minor)')
plt.xlabel('Mode')
plt.ylabel('Contagem')
plt.show()

# Distribuição de 'key'
plt.figure(figsize=(12, 6))
sns.countplot(data=spotify_data, x='key', order=spotify_data['key'].value_counts().index)
plt.title('Distribuição das Chaves (Keys)')
plt.xlabel('Key')
plt.ylabel('Contagem')
plt.xticks(rotation=45)
plt.show()