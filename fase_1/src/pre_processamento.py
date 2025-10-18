import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def limpeza(dados: pd.DataFrame) -> pd.DataFrame:
    '''
    Trata as informações que podem prejudicar o treinamento do modelo, como zeros inválidos.
    Foi considerado que as seguintes colunas devem sempre ter um valor maior do que zero:
    - Glicose
    - Pressão arterial
    - Espessura da pele
    - Insulina
    - IMC
    - Idade

    Parâmetros:
        dados: DataFrame com as informações que devem ser tratadas

    Retorno:
        DataFrame com as informações já tratadas
    '''
    print('\nIniciando a limpeza')

    # Removendo colunas com grande quantidade de valores inválidos (acima de 25%)
    # dados_limpeza.drop(columns=['Insulina', 'Espessura da pele'], axis=1)

    # Tratamento de zeros em colunas onde isso é inválido.
    colunas = ['Glicose', 'Pressão arterial', 'Espessura da pele', 'Insulina', 'IMC', 'Idade']
    dados[colunas] = dados[colunas].replace(0, np.nan)

    # Tratamento de nulos em colunas onde isso é inválido.
    for coluna in colunas:
        dados[coluna].fillna(dados[coluna].median()) # Preenchendo com mediana
        # dados_limpeza[coluna].fillna(dados_limpeza[coluna].mean(), inplace=True) # Preenchendo com média

        # Preenchendo com KNNImputer
        # imputer = KNNImputer()
        # dados_knn = pd.DataFrame(
        #     imputer.fit_transform(dados_limpeza[coluna].to_frame()),
        #     columns=[coluna]
        # )
        # dados_limpeza[coluna] = dados_knn[coluna]

    print('\nFinalizando a limpeza')

    return dados

def escalonamento(X_treino, X_teste):
    '''
    Ajusta os dados para a escala de cada coluna usando padronização, uma vez que há muitos outliers.

    Parâmetros:
        X_treino: dados de treino
        X_teste: dados de teste

    Retorno:
        X_treino escalonado
        X_teste escalonado
    '''

    print('\nIniciando o escalonamento')
    
    scaler = StandardScaler() 
    X_treino_escalado = scaler.fit_transform(X_treino)
    X_teste_escalado = scaler.transform(X_teste)
    
    X_treino_escalado_df = pd.DataFrame(X_treino_escalado, columns=X_treino.columns)
    X_teste_escalado_df = pd.DataFrame(X_teste_escalado, columns=X_teste.columns)

    print('\nFinalizando o escalonamento')

    return X_treino_escalado_df, X_teste_escalado_df