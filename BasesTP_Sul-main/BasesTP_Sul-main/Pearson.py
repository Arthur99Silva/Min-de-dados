import pandas as pd
import scipy.stats as stats

# Carregar o dataset
file_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/Dataset.csv"
df = pd.read_csv(file_path)

# Calcular a correlação de Pearson usando scipy.stats
correlacoes_pearson = {
    "Produção x Temperatura Média": stats.pearsonr(df["Produção (t)"], df["TemperaturaMedia C"])[0],
    "Produção x Precipitação Total": stats.pearsonr(df["Produção (t)"], df["PRECIPITACAO TOTAL, MENSAL (AUT)(mm)"])[0],
    "Produção x Vento Velocidade Média": stats.pearsonr(df["Produção (t)"], df["VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)"])[0],
}

# Calcular a correlação de Spearman usando scipy.stats
correlacoes_spearman = {
    "Produção x Temperatura Média": stats.spearmanr(df["Produção (t)"], df["Temperatura Media"])[0],
    "Produção x Precipitação Total": stats.spearmanr(df["Produção (t)"], df["PRECIPITACAO TOTAL, MENSAL (AUT)(mm)"])[0],
    "Produção x Vento Velocidade Média": stats.spearmanr(df["Produção (t)"], df["VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)"])[0],
}

# Exibir os resultados
print("Correlação de Pearson:")
print(correlacoes_pearson)

print("\nCorrelação de Spearman:")
print(correlacoes_spearman)
