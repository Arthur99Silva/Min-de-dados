import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

spotify_data = pd.read_csv('spotify-2023.csv', encoding='ISO-8859-1')

# Pré-processamento
# Convertendo a coluna 'streams' para numérico
spotify_data['streams'] = pd.to_numeric(spotify_data['streams'].str.replace(',', ''), errors='coerce')

# Características usadas para a classificação
features = ['danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 
            'instrumentalness_%', 'liveness_%', 'speechiness_%']

# Variável alvo: se a música está nas paradas de sucesso
target = 'in_shazam_charts'

spotify_data = spotify_data.dropna(subset=features + [target])

# Separar as variáveis
X = spotify_data[features]
y = spotify_data[target]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo de Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Fazer o predict
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')
print('Relatório de Classificação:')
print(classification_report(y_test, y_pred))

# Plot
importances = clf.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
plt.figure(figsize=(10, 6))
plt.title('Importância das Características - Random Forest')
plt.bar(range(len(importances)), [importances[i] for i in indices], align='center')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.xlabel('Características')
plt.ylabel('Importância')
plt.show()
