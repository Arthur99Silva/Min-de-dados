# Importar bibliotecas
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Carregar os dados do Spotify
spotify_data = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

# Discretizar as colunas numéricas para criar categorias (baixa, média, alta)
spotify_data['valence_cat'] = pd.cut(spotify_data['valence_%'], bins=[0, 33, 66, 100], labels=['low', 'medium', 'high'])
spotify_data['energy_cat'] = pd.cut(spotify_data['energy_%'], bins=[0, 33, 66, 100], labels=['low', 'medium', 'high'])
spotify_data['danceability_cat'] = pd.cut(spotify_data['danceability_%'], bins=[0, 33, 66, 100], labels=['low', 'medium', 'high'])

# Selecionar as colunas relevantes para mineração de regras
df_association = spotify_data[['valence_cat', 'energy_cat', 'danceability_cat']]

# Converter as colunas categóricas em variáveis dummies
df_dummies = pd.get_dummies(df_association)

# Aplicar o algoritmo Apriori para encontrar conjuntos frequentes
frequent_itemsets = apriori(df_dummies, min_support=0.1, use_colnames=True)

# Gerar regras de associação a partir dos conjuntos frequentes
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Exibir as primeiras regras
print(rules.head())

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.7, c=rules['lift'], cmap='viridis')
plt.colorbar(label='Lift')
plt.title('Regras de Associação: Suporte vs Confiança')
plt.xlabel('Support (Suporte)')
plt.ylabel('Confidence (Confiança)')
plt.grid(True)
plt.show()
