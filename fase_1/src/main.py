import pandas
import exploracao
import pre_processamento
# import predicao
from pathlib import Path
from download import Kaggle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import classification_report

print('Início do Tech Challenge - Fase 1')
print('Diagnóstico de diabetes')

# Download da base de dados
destino = Path().cwd() / 'fase_1' / 'data'
if not any(destino.iterdir()):
    Kaggle.download('mathchi/diabetes-data-set', str(destino))

# Carregamento da base de dados em um DataFrame do Pandas.
dados = pandas.read_csv(destino / 'diabetes.csv')

# Tradução dos nomes das colunas para facilitar a análise dos dados.
dados = dados.rename(columns={
    'Pregnancies': 'Gestações',
    'Glucose': 'Glicose',
    'BloodPressure': 'Pressão arterial',
    'SkinThickness': 'Espessura da pele',
    'Insulin': 'Insulina',
    'BMI': 'IMC',
    'DiabetesPedigreeFunction': 'Hereditariedade',
    'Age': 'Idade',
    'Outcome': 'Diagnóstico'
})

# Análise dos dados antes do pré-processamento
exploracao.analise_descritiva(dados)
exploracao.analise_grafica(dados)

# Pré-processamento dos dados
dados = pre_processamento.limpeza(dados)

X = dados.drop(columns=['Diagnóstico'])
y = dados['Diagnóstico']
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_treino_escalonado, X_teste_escalonado = pre_processamento.escalonamento(X_treino, X_teste)

# # Análise dos dados depois do pré-processamento
exploracao.analise_descritiva(dados)
exploracao.analise_grafica(dados)

# Precisa balancear os dados?

# Treino e avaliação dos modelos
modelos = {
    'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42),
    'Árvore de Descisão': DecisionTreeClassifier(
        criterion='gini', # Testar com os 3 critérios que existem aqui
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
    # 'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

# melhor_modelo = predicao.analisar_modelos(modelos, X_treino_balanceado, y_treino_balanceado, X_teste_escalado, y_teste)

# ----------------------------
# Avaliação final no conjunto de teste
# ----------------------------
# y_predito_final = melhor_modelo.predict(X_teste_escalado)
# print('\n' + classification_report(y_teste, y_predito_final))

# Ainda é preciso interpretar os resultados usando feature importance e SHAP.

# Discussão dos resultados: 
# - O modelo pode ser usado na prática? Como?

print('\nFim do Tech Challenge - Fase 1')