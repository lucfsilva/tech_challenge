import kagglehub
import traducao
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def carregar(origem: str) -> pd.DataFrame:
    '''
    Carrega um Dataset do Kaggle em um DataFrame do Pandas.

    Parâmetros:
        origem(str): identificador do Dataset no Kaggle.

    Retorno:
        DataFrame com o conteúdo do Dataset do Kaggle.
    '''

    print('\nIniciando o carregamento dos dados')

    print('')
    endereco_de_origem = kagglehub.dataset_download(handle=origem, force_download=True)
    diretorio_de_origem = Path(endereco_de_origem).resolve()

    lista_dados_csv = []

    for item in diretorio_de_origem.iterdir():
        if item.is_file() and item.suffix.lower() == '.csv':
            lista_dados_csv.append(pd.read_csv(item))

    dados = pd.concat(lista_dados_csv, axis=0, ignore_index=True)

    # dados = pd.read_csv(Path().cwd() / 'fase_1' / 'data' / 'diabetes.csv') # Para trabalhar com o arquivo localmente.
    
    print('\nFinalizando o carregamento dos dados')
    
    return dados

def analise_descritiva(dados: pd.DataFrame):
    '''
    Descreve características relevantes dos dados, como:
    - Estrutura
    - Zeros inválidos

    Parâmetros:
        dados: DataFrame que deve ser analisado
    '''

    print('\nIniciando a análise descritiva dos dados')

    dados_traduzidos = traducao.traduzir(dados)

    print('\nEstrutura:')
    print(dados_traduzidos.info())

    # print('\nResumo estatístico:')
    # print(dados.describe())

    print('\nZeros inválidos:')
    dados_invalidos = dados_traduzidos[['Glicose', 'Pressão', 'Espessura da pele', 'Insulina', 'IMC', 'Idade']] <= 0
    print((dados_invalidos.sum() / len(dados_traduzidos) * 100).round(2).map(lambda x: f'{x:.2f} %'))

    # print('\nPrimeiras linhas:')
    # print(dados.head())

    print('\nFinalizando a análise descritiva dos dados')

def analise_grafica(dados: pd.DataFrame):
    '''
    Mostra gráficos com características relevantes dos dados, como:
    - Distribuição do diagnóstico
    - Boxplot das características.
    - Mapa de calor entre características e diagnóstico.
    - Ranking de correlação entre características e diagnóstico.

    Parâmetros:
        dados: DataFrame que deve ser analisado
    '''

    print('\nIniciando a análise gráfica dos dados')

    dados_traduzidos = traducao.traduzir(dados)

    # Distribuição do diagnóstico
    contagens = dados_traduzidos['Diagnóstico'].value_counts().sort_index()

    plt.figure(figsize=(5, 5))
    plt.pie(
        contagens,
        labels=['Não diabético', 'Diabético'],
        autopct='%1.2f%%',
        startangle=90,
        colors=sns.color_palette('Set2')
    )
    plt.title('Diagnósticos')
    plt.show()

    # Boxplot das características
    dados_sem_diagnostico = dados_traduzidos.drop('Diagnóstico', axis=1)

    plt.figure(figsize=(10,8))
    plt.suptitle('Características', fontsize=14)
    for i, coluna in enumerate(dados_sem_diagnostico.columns):
        plt.subplot(3, 3, i+1)
        sns.boxplot(y=coluna, data=dados_sem_diagnostico, color='skyblue')
        plt.title(f'{coluna}')
        plt.ylabel('')
    plt.tight_layout()
    plt.show()

    # Proporção de outliers das característica, usando IQR
    # fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    # axes = axes.flatten()

    # for i, coluna in enumerate(dados_sem_diagnostico.columns):
    #     Q1 = dados_sem_diagnostico[coluna].quantile(0.25)
    #     Q3 = dados_sem_diagnostico[coluna].quantile(0.75)
    #     IQR = Q3 - Q1
    #     limite_inferior = Q1 - 1.5 * IQR
    #     limite_superior = Q3 + 1.5 * IQR

    #     outliers = ((dados_sem_diagnostico[coluna] < limite_inferior) | 
    #                 (dados_sem_diagnostico[coluna] > limite_superior))

    #     num_outliers = outliers.sum()
    #     num_normais = len(outliers) - num_outliers

    #     eixos = axes[i]
    #     eixos.pie(
    #         [num_normais, num_outliers],
    #         labels=['Normais', 'Outliers'],
    #         autopct='%1.1f%%',
    #         colors=['skyblue', 'salmon'],
    #         startangle=90
    #     )
    #     eixos.set_title(coluna)

    # for j in range(i + 1, len(axes)):
    #     fig.delaxes(axes[j])

    # plt.suptitle('Outliers', fontsize=14)
    # plt.tight_layout()
    # plt.show()

    # Mapa de calor entre características e diagnóstico
    matriz_de_correlacao = dados_traduzidos.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(matriz_de_correlacao, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de calor')
    plt.tight_layout()
    plt.show()

    # Ranking da correlação entre características e diagnóstico
    correlacao = matriz_de_correlacao['Diagnóstico'].drop('Diagnóstico')
    coluna_X = 'Característica'
    coluna_y = 'Correlação'
    ranking = correlacao.abs().sort_values(ascending=False)
    ranking_df = ranking.reset_index()
    ranking_df.columns = [coluna_X, coluna_y]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=ranking_df, x=coluna_X, y=coluna_y, color='skyblue')
    plt.title('Ranking de correlação', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print('\nFinalizando a análise gráfica dos dados')