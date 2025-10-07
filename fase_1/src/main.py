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

X_treino_escalonado, X_teste_escalonado = pre_processamento.escalonar(RobustScaler(), X_treino, X_teste)
# também pode usar pre_processamento.escalonar com StandardScaler() ou MinMaxScaler() no lugar de RobustScaler()

X_treino_balanceado, y_treino_balanceado = pre_processamento.balancear(X_treino_escalonado, y_treino)