import pandas
import exploracao
import limpeza
import pre_processamento
import predicao
from pathlib import Path
from download import Kaggle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

print('Início do Tech Challenge - Fase 1')
print('Diagnóstico de diabetes\n')

# ----------------------------
# Carregamento da base de dados e definição de características importantes
# ----------------------------
destino = Path().cwd() / 'fase_1' / 'data'
if not any(destino.iterdir()):
    Kaggle.download('mathchi/diabetes-data-set', str(destino))

dados = pandas.read_csv(destino / 'diabetes.csv')

# ----------------------------
# Análise dos dados antes da aplicação da limpeza e tratamento de outliers
# ----------------------------
exploracao.analise_descritiva(dados)
exploracao.analise_grafica(dados)

# colunas_positivas = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

# ----------------------------
# Limpeza dos dados e tratamento de outliers
# ----------------------------
# dados = limpeza.limpar(dados, colunas_positivas)
# dados = limpeza.limpar_dados(dados, colunas_positivas)
# dados = pre_processamento.tratar_outliers(dados)

# ----------------------------
# Análise dos dados depois da aplicação da limpeza e tratamento de outliers
# ----------------------------
# colunas_originais = [coluna for coluna in dados.columns if not coluna.endswith('_outlier')]
# dados_pos_limpeza = dados[colunas_originais]
# exploracao.analisar_dados(dados_pos_limpeza, coluna_diagnostico)

# ----------------------------
# Separação de treino e teste e executar pré processamento
# ----------------------------
# X = dados.drop(columns=[coluna_diagnostico])
# y = dados[coluna_diagnostico]

# X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# X_treino_escalado, X_teste_escalado = pre_processamento.tratar_escala(RobustScaler(), X_treino, X_teste)
# Pode usar pre_processamento.escalar com RobustScaler(), StandardScaler() ou MinMaxScaler().

# X_treino_balanceado, y_treino_balanceado = pre_processamento.balancear(X_treino_escalado, y_treino)

# ----------------------------
# Treino e avaliação dos modelos
# ----------------------------
# modelos = {
#     'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42),
#     'Árvore de Descisão': DecisionTreeClassifier(
#         criterion='gini',
#         max_depth=5,
#         min_samples_split=10,
#         min_samples_leaf=5,
#         random_state=42),
#     'KNN': KNeighborsClassifier(n_neighbors=5),
#     'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
#     'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
# }

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