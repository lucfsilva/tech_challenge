import kagglehub
from pathlib import Path
import shutil

diretorio_de_destino = Path(Path().cwd() / 'fase_1' / 'data')
if not any(diretorio_de_destino.iterdir()):
    endereco_de_origem = kagglehub.dataset_download('mathchi/diabetes-data-set')
    diretorio_de_origem = Path(endereco_de_origem)

    for item in diretorio_de_origem.iterdir():
        if item.is_file():
            shutil.move(item, diretorio_de_destino)