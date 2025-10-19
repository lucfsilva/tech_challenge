import exploracao
import pre_processamento
import modelagem
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

print('\nInício do Tech Challenge - Fase 1: diagnóstico de doenças')

dados = exploracao.carregar('mathchi/diabetes-data-set')
exploracao.analise_descritiva(dados)
exploracao.analise_grafica(dados)

dados = pre_processamento.limpar(dados)

X_dados = dados.drop(columns=['Outcome'], axis=1)
y_dados = dados['Outcome']
X_treino, X_teste, y_treino, y_teste = train_test_split(X_dados, y_dados, test_size=0.2, random_state=42, stratify=y_dados)

X_treino_padronizado, X_teste_padronizado = pre_processamento.padronizar(X_treino, X_teste)
# X_treino_balanceado, y_treino_balanceado = pre_processamento.balancear(X_treino_padronizado, y_treino) # Não aplicado pois o melhor modelo não precisou

# X_total = pd.concat([X_treino_padronizado, X_teste_padronizado], axis=0)
# y_total = pd.concat([y_treino, y_teste], axis=0)
# dados = X_total.copy()
# dados['Outcome'] = y_total.values
# exploracao.analise_descritiva(dados)
# exploracao.analise_grafica(dados)

modelos = [
    LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
        ),
    DecisionTreeClassifier(
        class_weight='balanced',
        criterion='entropy',
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42),
    KNeighborsClassifier(
        n_neighbors=5),
    RandomForestClassifier(
        class_weight='balanced',
        n_estimators=200,
        random_state=42),
    GradientBoostingClassifier(
        n_estimators=200,
        random_state=42)
]

modelos_treinados = modelagem.treinar(modelos, X_treino_padronizado, y_treino)
# modelos_treinados = modelagem.treinar(modelos, X_treino_balanceado, y_treino_balanceado)
melhor_modelo = modelagem.avaliar(modelos_treinados, X_teste_padronizado, y_teste)
modelagem.analisar(melhor_modelo, X_treino_padronizado)

print('\nFim do Tech Challenge - Fase 1')