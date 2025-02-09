import pandas as pd

# Carregar o arquivo CSV
file_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/TempCompleto_Tratado.csv"
df = pd.read_csv(file_path)

# Verificar se há duplicatas exatas
duplicatas = df.duplicated(subset=["Data", "Estado"], keep=False)
num_duplicatas = duplicatas.sum()

# Se houver duplicatas, agrupar os dados tirando a média das medições repetidas
if num_duplicatas > 0:
    df_unique = df.groupby(["Data", "Estado"], as_index=False).mean(numeric_only=True)
else:
    df_unique = df.copy()

# Salvar o dataset corrigido
output_file_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/TempCompleto_Corrigido.csv"
df_unique.to_csv(output_file_path, index=False)