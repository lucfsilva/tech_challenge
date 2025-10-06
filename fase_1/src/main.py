from pathlib import Path
from download import Kaggle
import pandas as pd
import exploracao

print('Tech Challenge - Fase 1')
print('Diagnóstico de diabetes\n')

destino = Path().cwd() / 'fase_1' / 'data'
if not any(destino.iterdir()):
    Kaggle.download('mathchi/diabetes-data-set', str(destino))

dados = pd.read_csv(destino / 'diabetes.csv')

exploracao.analisar_dados(dados, 'Outcome')
