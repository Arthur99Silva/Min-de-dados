import pandas as pd
import numpy as np

# Carregar a tabela fornecida
file_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/New_TempSul.csv"
df = pd.read_csv(file_path)

# Converter colunas numéricas que estão como string com vírgulas para ponto
cols_to_convert = [
    "PRECIPITACAO TOTAL, MENSAL (AUT)(mm)",
    "PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)",
    "VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)"
]

for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Identificar colunas numéricas para tratamento de NaN e outliers
numeric_cols = ["TemperaturaMedia C"] + cols_to_convert

# Preencher valores ausentes com estratégias apropriadas
df["TemperaturaMedia C"].fillna(df["TemperaturaMedia C"].median(), inplace=True)
df["PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)"].fillna(df["PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)"].median(), inplace=True)
df["PRECIPITACAO TOTAL, MENSAL (AUT)(mm)"].fillna(df["PRECIPITACAO TOTAL, MENSAL (AUT)(mm)"].mean(), inplace=True)
df["VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)"].fillna(df["VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)"].mean(), inplace=True)

# Função para substituir outliers pelo limite máximo/mínimo usando IQR
def replace_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

# Aplicar substituição de outliers usando IQR
df_cleaned = replace_outliers(df, numeric_cols)

# Salvar o CSV tratado
output_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/ClimaKNN.csv"
df_cleaned.to_csv(output_path, index=False)

print(f"Arquivo tratado salvo em: {output_path}")
