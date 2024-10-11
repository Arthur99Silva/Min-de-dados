import pandas as pd

# Carregar o conjunto de dados
spotify_data = pd.read_csv("spotify-2023.csv", encoding='ISO-8859-1')

# 1. Identificação de Valores Ausentes
missing_values = spotify_data.isnull().sum()
print("Valores Ausentes por Coluna:")
print(missing_values)

# 2. Tratamento de Valores Ausentes

# Tratando valores ausentes na coluna 'key' (preenchendo com 'Unknown')
spotify_data['key'] = spotify_data['key'].fillna('Unknown')

# Tratando valores ausentes na coluna 'in_shazam_charts' (preenchendo com 0)
spotify_data['in_shazam_charts'] = spotify_data['in_shazam_charts'].fillna(0)

# 3. Conversão e Padronização dos Dados

# Remover separadores de milhar e converter a coluna 'streams' para float
spotify_data['streams'] = pd.to_numeric(spotify_data['streams'].str.replace(',', ''), errors='coerce')

# Verificando o tipo das colunas para garantir que estão corretos
print("\nTipos de Dados Após Conversão:")
print(spotify_data.dtypes)

# 4. Verificação Final de Valores Ausentes
missing_values_after_treatment = spotify_data.isnull().sum()
print("\nValores Ausentes Após Tratamento:")
print(missing_values_after_treatment)

# 5. Exibindo as primeiras linhas para garantir que os dados foram processados corretamente
print("\nPrimeiras Linhas dos Dados Processados:")
print(spotify_data.head())
