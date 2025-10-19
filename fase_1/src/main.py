import exploracao
import pre_processamento
import modelagem
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

print('Início do Tech Challenge - Fase 1')
print('Diagnóstico de diabetes')

dados = exploracao.carregar('mathchi/diabetes-data-set')
exploracao.analise_descritiva(dados)
exploracao.analise_grafica(dados)

dados = pre_processamento.limpar(dados)

X_dados = dados.drop(columns=['Outcome'], axis=1)
y_dados = dados['Outcome']
X_treino, X_teste, y_treino, y_teste = train_test_split(X_dados, y_dados, test_size=0.2, random_state=42, stratify=y_dados)

X_treino_escalonado, X_teste_escalonado = pre_processamento.escalonar(X_treino, X_teste)
# X_treino_balanceado, y_treino_balanceado = pre_processamento.balancear(X_treino_escalonado, y_treino) # Melhora o resultado de Regressão Logística e KNN, mas não o suficiente para superar a árvore de descisão.

X_total = pd.concat([X_treino_escalonado, X_teste_escalonado], axis=0)
y_total = pd.concat([y_treino, y_teste], axis=0)
dados = X_total.copy()
dados['Outcome'] = y_total.values
exploracao.analise_descritiva(dados)
exploracao.analise_grafica(dados)

# Treino e avaliação dos modelos
modelos = {
    'Regressão Logística': LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
        ),
    'Árvore de Descisão': DecisionTreeClassifier(
        class_weight='balanced', # Melhora o resultado final, mas desajusta o Recall entre 0 e 1
        criterion='entropy',
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42),
    'KNN': KNeighborsClassifier(
        n_neighbors=5)
    # 'Árvore Aleatória': RandomForestClassifier(
    #     class_weight='balanced',
    #     n_estimators=200,
    #     random_state=42),
    # 'Gradient Boosting': GradientBoostingClassifier(
    #     n_estimators=200,
    #     random_state=42)
}

melhor_modelo = modelagem.avaliar(modelos, X_treino_escalonado, y_treino, X_teste_escalonado, y_teste) # Sem balanceamento
# melhor_modelo = modelagem.avaliar(modelos, X_treino_balanceado, y_treino_balanceado, X_teste_escalonado, y_teste) # Com balanceamento

# Ainda é preciso interpretar os resultados usando feature importance e SHAP.

# Discussão dos resultados: 
# - O modelo pode ser usado na prática? Como?

print('\nFim do Tech Challenge - Fase 1')