import pandas as pd
from pathlib import Path

caminho_do_dataframe = Path().cwd() / 'fase_1' / 'data' / 'diabetes.csv'
dados = pd.read_csv(str(caminho_do_dataframe))

print(dados.shape)
dados.info()

