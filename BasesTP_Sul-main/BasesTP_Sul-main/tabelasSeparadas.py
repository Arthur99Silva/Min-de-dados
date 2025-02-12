import pandas as pd

# Carregar os dados do CSV principal
comparacao_df = pd.read_csv("/home/arthurantunes/Documentos/Min-de-dados-main/BasesTP_Sul-main/BasesTP_Sul-main/comparacao_modelos.csv")

# Criar três tabelas separadas
producao_df = comparacao_df[["Métrica", "Produção - Random Forest", "Produção - Gradient Boosting"]]
rendimento_df = comparacao_df[["Métrica", "Rendimento Médio - Random Forest", "Rendimento Médio - Gradient Boosting"]]
area_colhida_df = comparacao_df[["Métrica", "Área Colhida - Random Forest", "Área Colhida - Gradient Boosting"]]

# Salvar cada tabela separadamente
producao_df.to_csv("comparacao_producao.csv", index=False)
rendimento_df.to_csv("comparacao_rendimento.csv", index=False)
area_colhida_df.to_csv("comparacao_area_colhida.csv", index=False)

print("Tabelas salvas com sucesso!")
