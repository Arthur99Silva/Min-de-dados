import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

# Carregar o arquivo
df = pd.read_csv("C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/Temp_Estados.csv")

# Verificar quantos valores NaN existem no dataset antes da imputação
print("Valores NaN antes da imputação:")
print(df.isna().sum())

# Aplicar KNN Imputer para preencher os valores NaN, considerando os vizinhos mais próximos
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')  # Pesos baseados na distância

# Selecionar apenas colunas numéricas para imputação
numerical_cols = ['Temperatura Media', 'Precipitacao Total', 'Pressao Atmosferica Media', 'Vento Velocidade Media']
df[numerical_cols] = knn_imputer.fit_transform(df[numerical_cols])

# Arredondar os valores para uma casa decimal
df[numerical_cols] = df[numerical_cols].round(1)
# Adicionar um '0' no início das datas que começam com os padrões especificados
df['Data'] = df['Data'].astype(str).apply(lambda x: '0' + x if x[:3] in ['120', '220', '320', '420', '520', '620', '720', '820', '920'] else x)

# Verificar se ainda existem valores NaN após a imputação
print("Valores NaN após a imputação:")
print(df.isna().sum())

# Salvar o arquivo atualizado
df.to_csv("C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/Temp_KNN_Estados.csv", index=False)
