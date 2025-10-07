from pathlib import Path
from download import Kaggle
import pandas as pd
import exploracao
import limpeza
import pre_processamento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

print('Tech Challenge - Fase 1')
print('Diagnóstico de diabetes\n')

destino = Path().cwd() / 'fase_1' / 'data'
if not any(destino.iterdir()):
    Kaggle.download('mathchi/diabetes-data-set', str(destino))

dados = pd.read_csv(destino / 'diabetes.csv')
coluna_target = 'Outcome'
colunas_positivas = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

exploracao.analisar_dados(dados, coluna_target)
dados = limpeza.limpar_dados(dados, colunas_positivas)
dados, sufixo_outlier = pre_processamento.tratar_outliers(dados)
exploracao.analisar_correlacao(dados, coluna_target, sufixo_outlier)

X = dados.drop(columns=[coluna_target])
y = dados[coluna_target]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# X_train_regressao_logistica = pre_processamento.escalonar_para_regressao_logistica(X_train, 'Outcome')
# X_train_para_knn = pre_processamento.escalonar_para_knn(X_train, 'Outcome')
# X_train_para_outliers = pre_processamento.escalonar_para_outliers(X_train, 'Outcome')
    # X_train_scaled_df, X_test_scaled_df = escalonar(StandardScaler(), X_train, X_test)
    # X_train_scaled_df, X_test_scaled_df = escalonar(MinMaxScaler(), X_train, X_test)
    # X_train_scaled_df, X_test_scaled_df = escalonar(RobustScaler(), X_train, X_test)

# dados = pre_processamento.balancear(X_train, 'Outcome') # -> Precisa ser feito após o escalonamento