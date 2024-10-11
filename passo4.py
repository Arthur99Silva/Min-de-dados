import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Carregar o conjunto de dados
spotify_data = pd.read_csv("spotify-2023.csv", encoding='ISO-8859-1')

# Convertendo a coluna 'streams' para numérico caso ainda não esteja no formato correto
spotify_data['streams'] = pd.to_numeric(spotify_data['streams'].str.replace(',', ''), errors='coerce')

# 1. Seleção de Variáveis Relevantes para o Clustering
features = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 
            'instrumentalness_%', 'liveness_%', 'speechiness_%']

X = spotify_data[features]

# 2. Normalização dos Dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Aplicando o Algoritmo K-means
# Escolhendo 4 clusters como exemplo
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Adicionando o rótulo dos clusters ao DataFrame original
spotify_data['cluster'] = kmeans.labels_

# 4. Visualização dos Grupos (Usando PCA para reduzir a dimensionalidade a 2D para visualização)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=spotify_data['cluster'], palette='viridis')
plt.title('Visualização dos Clusters de Músicas (PCA 2D)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster')
plt.show()

# 5. Análise das Características de Cada Cluster
cluster_summary = spotify_data.groupby('cluster')[features].mean()
print("\nResumo das Características por Cluster:")
print(cluster_summary)
