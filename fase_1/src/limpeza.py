import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def limpar_dados(
        dados, 
        colunas_invalidas, 
        limte_inferior_imputacao_simples=0, 
        limite_superior_imputacao_simples=20, 
        limite_skew_imputacao_simples=0.5, 
        limite_inferior_imputacao_avancada=20, 
        limite_superir_imputacao_avancada=40):
    '''
    Limpa os dados, trocando valores ausentes ou zerados nas colunas_invalidas por substitutos adequados. 
    Usa imputação simples para ausências estabelecidas entre o limte_inferior_imputacao_simples e limite_superior_imputacao_simples
    Usa imputação avançada para ausências entre o limite_inferior_imputacao_avancada e limite_superir_imputacao_avancada
    Remove colunas com ausências maiores que o limite_superir_imputacao_avancada

    Parâmetros:
        dados (DataFrame): tabela com informações que devem ser limpas.
        colunas_invalidas (list): lista de colunas que não podem ter valor zero ou nulo.
        limte_inferior_imputacao_simples (float): porcentagem mínima de ausências para se aplicar imputação simples.
        limite_superior_imputacao_simples (float): porcentagem máxima de ausências para se aplicar a imputação simples.
        limite_skew_imputacao_simples (float): valor de skewness acima do qual se usa mediana em imputação simples
        limite_inferior_imputacao_avancada (float): porcentagem mínima de ausências para se aplicar a imputação avançada.
        limite_superir_imputacao_avancada (float): porcentagem máxima de ausências para se aplicar a imputação avançada.

    Retorno:
        DataFrame: Tabela com informações após limpeza.
    '''
    print('Iniciando a limpeza dos dados')

    # ----------------------------
    # Identificação de valores ausentes
    # ----------------------------
    print('\nValores ausentes por coluna (antes do tratamento):')
    print(dados.isnull().sum())

    # ----------------------------
    # Substituir valores 0 inválidos por NaN
    # ----------------------------
    for coluna in colunas_invalidas:
        dados[coluna] = dados[coluna].replace(0, np.nan)

    print('\nValores ausentes após substituir zeros inválidos por NaN:')
    print(dados.isnull().sum())

    # ----------------------------
    # Análise das ausências e imputação de valores usando a técnica mais adequada
    # ----------------------------
    def imputacao_simples(series_coluna):
        if series_coluna.dtype in ['int64', 'float64']:
            skew_value = series_coluna.skew()
            if abs(skew_value) > limite_skew_imputacao_simples:
                imputacao = 'Mediana'
                series_coluna.fillna(series_coluna.median(), inplace=True)
            else:
                imputacao = 'Média'
                series_coluna.fillna(series_coluna.mean(), inplace=True)
        else:
            imputacao = 'Moda'
            series_coluna.fillna(series_coluna.mode()[0], inplace=True)

        return 'Imputação simples', imputacao
    
    def imputacao_avancada(series_coluna):
        imputer = KNNImputer(n_neighbors=5)
        dados_knn = pd.DataFrame(
            imputer.fit_transform(series_coluna.to_frame()),
            columns=[series_coluna.name]
        )

        series_coluna = dados_knn[series_coluna.name]

        return series_coluna, 'Imputação avançada', 'KNN'

    total = len(dados)
    resumo = pd.DataFrame(columns=['% Ausentes', 'Estratégia', 'Imputação'])
    for coluna in colunas_invalidas:
        porcentagem = dados[coluna].isnull().sum() / total * 100
        resumo.loc[coluna, '% Ausentes'] = round(porcentagem, 2)

        if porcentagem > limte_inferior_imputacao_simples and porcentagem <= limite_superior_imputacao_simples:
            estrategia, imputacao = imputacao_simples(dados[coluna])
        elif porcentagem > limite_inferior_imputacao_avancada and porcentagem <= limite_superir_imputacao_avancada:
            dados[coluna], estrategia, imputacao = imputacao_avancada(dados[coluna])
        elif porcentagem > limite_superir_imputacao_avancada:
            estrategia = 'Descartar coluna'
            imputacao = 'Exclusão'
            dados = dados.drop(columns=[coluna])
        else:
            estrategia = 'Valor não tratado'
            imputacao = 'Nenhuma'

        resumo.loc[coluna, 'Estratégia'] = estrategia
        resumo.loc[coluna, 'Imputação'] = imputacao    

    print('\nResumo das ações de limpeza')
    print(resumo)

    print('\nValores ausentes após imputação:')
    print(dados.isnull().sum())

    # ----------------------------
    # Conferir estatísticas após limpeza
    # ----------------------------
    print('\nResumo estatístico após limpeza:')
    print(dados.describe())

    print('\nFinalizando a limpeza dos dados')

    return dados