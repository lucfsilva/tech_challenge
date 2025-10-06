from pathlib import Path
import kagglehub
import shutil

class Kaggle:
    @staticmethod
    def download(origem: str, destino: str) -> Path:
        """
        Faz o download de um dataset do Kaggle usando kagglehub e move os arquivos para o destino.
        
        Args:
            origem (str): ID do dataset no Kaggle (ex: "mathchi/diabetes-data-set")
            destino (str): Caminho da pasta de destino

        Returns:
            Path: Caminho da pasta de destino final
        """
        # Diretório de destino
        diretorio_de_destino = Path(destino).resolve()
        diretorio_de_destino.mkdir(parents=True, exist_ok=True)

        # Baixa o dataset do Kaggle
        endereco_de_origem = kagglehub.dataset_download(handle=origem,force_download=True)
        diretorio_de_origem = Path(endereco_de_origem).resolve()

        # Move os arquivos baixados
        for item in diretorio_de_origem.iterdir():
            destino_item = diretorio_de_destino / item.name
            if destino_item.exists():
                # remove antes de mover, para evitar erro de conflito
                if destino_item.is_file():
                    destino_item.unlink()
                elif destino_item.is_dir():
                    shutil.rmtree(destino_item)

            shutil.move(str(item), str(diretorio_de_destino))

        return diretorio_de_destino