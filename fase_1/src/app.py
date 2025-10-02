from download import Kaggle
from pathlib import Path

origem = 'mathchi/diabetes-data-set'
destino = Path().cwd().parent / 'data'
Kaggle.download(origem, str(destino))