import pandas as pd

# Criar dataframe com os resultados
comparacao_df = pd.DataFrame({
    "Métrica": ["Erro Médio Absoluto (MAE)", "Erro Quadrático Médio (MSE)", "Raiz do Erro Quadrático Médio (RMSE)", "Coeficiente de Determinação (R²)"],
    "Produção - Random Forest": [10.28, 20255.28, 142.32, 0.9999999995],
    "Produção - Gradient Boosting": [3647.75, 35065427.73, 5921.60, 0.9999991286],
    "Rendimento Médio - Random Forest": [5.06, 1192.09, 34.52, 0.99983],
    "Rendimento Médio - Gradient Boosting": [63.09, 8337.74, 91.31, 0.99884],
    "Área Colhida - Random Forest": [861.77, 93156647.25, 9651.77, 0.99997947],
    "Área Colhida - Gradient Boosting": [7052.53, 182124328.36, 13495.34, 0.99995986]
})

# Salvar em um arquivo CSV
comparacao_df.to_csv("comparacao_modelos.csv", index=False)
print("Arquivo 'comparacao_modelos.csv' salvo com sucesso!")
