import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data  # Features (características)
y = iris.target  # Labels (rótulos das classes)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definindo o modelo SVM
model = SVC()

# Realizando Grid Search para otimizar os hiperparâmetros
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['linear', 'rbf']  # Testando kernels linear e RBF
}

grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_search.fit(X_train, y_train)

# Melhor configuração de parâmetros
print(f"Melhores parâmetros: {grid_search.best_params_}")

# Usando o melhor modelo encontrado pelo GridSearch
best_model = grid_search.best_estimator_

# Avaliar o modelo usando validação cruzada
scores = cross_val_score(best_model, X, y, cv=5)  # Validação cruzada com 5 folds
print(f'Acurácia média da validação cruzada: {scores.mean():.2f}')

# Treinar o modelo com o melhor conjunto de parâmetros
best_model.fit(X_train, y_train)

# Predição no conjunto de teste
y_pred = best_model.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

# Resultados finais
print(f'Acurácia no conjunto de teste: {accuracy:.2f}')
print('Relatório de Classificação:')
print(report)
