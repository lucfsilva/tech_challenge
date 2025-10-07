import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats import zscore

def remover_duplicados(dados):
    '''
    Remove os registros que possuem mais de uma cópia idêntica nos dados.

    Parâmetros:
        dados: conjunto de informações que devem ser tratadas.

    Retorno:
        dados: conjunto de informações após tratamento.
    '''

    duplicados = dados.duplicated().sum()
    print(f"\nNúmero de registros duplicados: {duplicados}")

    if duplicados > 0:
        dados = dados.drop_duplicates()

        duplicados = dados.duplicated().sum()
        print(f"Número de registros duplicados após remoção: {duplicados}")

    return dados

def imputacao_simples(series_coluna, limite_skew=0.5):
    '''
    Preenche uma coluna usando as seguintes regras:
    - Se a coluna for numérica e seu skew_value for maior que limite_skew, preenche com a mediana.
    - Se a coluna for numérica e seu skew_value for menor que limite_skew, preenche com a média.
    - Se a coluna não for numérica, preenche com a moda.

    Parâmetros:
        series_coluna: coluna que deve ser tratada
        limite_skew: valor usado como limite entre mediana e média

    Retorno:
        series_coluna: coluna após tratamento
        str: tipo de imputação feita ('Mediana', 'Media' ou 'Moda')
    '''

    if series_coluna.dtype in ['number']:
        skew_value = series_coluna.skew()
        if abs(skew_value) > limite_skew:
            imputacao = 'Mediana'
            series_coluna.fillna(series_coluna.median(), inplace=True)
        else:
            imputacao = 'Média'
            series_coluna.fillna(series_coluna.mean(), inplace=True)
    else:
        imputacao = 'Moda'
        series_coluna.fillna(series_coluna.mode()[0], inplace=True)

    return series_coluna, imputacao

def imputacao_avancada(series_coluna, n_vizinhos=5):
    '''
    Preenche uma coluna usando KNNImputer

    Parâmetros:
        series_coluna: coluna que deve ser tratada
        n_vizinhos: quantidade de vizinhos próximos que o KNNImputer deve considerar.

    Retorno:
        series_coluna: coluna após tratamento
        str: tipo de imputação feita ('KNN')
    '''

    imputer = KNNImputer(n_neighbors=n_vizinhos)
    dados_knn = pd.DataFrame(
        imputer.fit_transform(series_coluna.to_frame()),
        columns=[series_coluna.name]
    )

    series_coluna = dados_knn[series_coluna.name]

    return series_coluna, 'KNN'

def tratar_ausencias(
        dados, 
        colunas_invalidas, 
        limte_inferior_imputacao_simples=0, 
        limite_superior_imputacao_simples=20, 
        limite_skew_imputacao_simples=0.5, 
        limite_inferior_imputacao_avancada=20, 
        limite_superir_imputacao_avancada=40,
        n_vizinhos_knn_imputacao_avancada=5):
    '''
    Aplica os seguintes tratamentos para as colunas_invalidas que estejam zeradas ou não preenchidas:
    - Imputação simples para ausências entre o limte_inferior_imputacao_simples e limite_superior_imputacao_simples.
    - Imputação avançada para ausências entre o limite_inferior_imputacao_avancada e limite_superir_imputacao_avancada.
    - Remoção da coluna para ausências superiores ao limite_superir_imputacao_avancada.

    Parâmetros:
        dados: conjunto de informações que devem ser tratadas.
        colunas_invalidas: lista de colunas que não podem ter valor zero ou nulo.
        limte_inferior_imputacao_simples: porcentagem mínima de ausências para se aplicar imputação simples.
        limite_superior_imputacao_simples: porcentagem máxima de ausências para se aplicar a imputação simples.
        limite_skew_imputacao_simples: valor de skew de uma coluna acima do qual se usa mediana em imputação simples
        limite_inferior_imputacao_avancada: porcentagem mínima de ausências para se aplicar a imputação avançada.
        limite_superir_imputacao_avancada: porcentagem máxima de ausências para se aplicar a imputação avançada.
        n_vizinhos_knn_imputacao_avancada: quantidade de vizinhos próximos que o KNNImputer deve considerar na imputação avançada

    Retorno:
        dados: conjunto de informações após tratamento.
    '''

    # ----------------------------
    # Substituir valores 0 inválidos por NaN
    # ----------------------------
    for coluna in colunas_invalidas:
        dados[coluna] = dados[coluna].replace(0, np.nan)

    print('\nValores ausentes por coluna:')
    print(dados.isnull().sum())

    # ----------------------------
    # Análise das ausências e imputação de valores usando a técnica mais adequada
    # ----------------------------
    total = len(dados)
    resumo = pd.DataFrame(columns=['% Ausentes', 'Estratégia', 'Imputação'])
    for coluna in colunas_invalidas:
        porcentagem = dados[coluna].isnull().sum() / total * 100
        resumo.loc[coluna, '% Ausentes'] = round(porcentagem, 2)

        if porcentagem > limte_inferior_imputacao_simples and porcentagem <= limite_superior_imputacao_simples:
            estrategia = 'Imputação simples'
            dados[coluna], imputacao = imputacao_simples(dados[coluna], limite_skew_imputacao_simples)
        elif porcentagem > limite_inferior_imputacao_avancada and porcentagem <= limite_superir_imputacao_avancada:
            estrategia = 'Imputação avançada'
            dados[coluna], imputacao = imputacao_avancada(dados[coluna], n_vizinhos_knn_imputacao_avancada)
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
    
    return dados

def tratar_outliers(dados, limite_skew=0.5, limite_zscore=3):
    '''
    Aplica os seguintes tratamentos para outliers:
    - Método Z-score para coluna com skew menor que limite_skew
    - Método IQR para coluna com skew maior que limite_skew

    Parâmetros:
        dados: conjunto de informações que devem ser tratadas.
        limite_skew: valor usado como limite entre os métodos Z-score e IQR.
        limite_zscore: valor de zscore que define uma coluna como outlier ou não no método Z-score.

    Retorno:
        dados: conjunto de informações após tratamento.
    '''

    colunas_numericas = dados.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for coluna in colunas_numericas:
        skew_value = dados[coluna].skew()
        
        if abs(skew_value) <= limite_skew:
            # ----------------------------
            # Método Z-score
            # ----------------------------
            z_scores = zscore(dados[coluna], nan_policy='omit')
            dados[coluna + '_outlier'] = (np.abs(z_scores) > limite_zscore).astype(int)
            
            media = dados[coluna].mean()
            desvio = dados[coluna].std()
            limite_inferior = media - limite_zscore * desvio
            limite_superior = media + limite_zscore * desvio
            
            dados[coluna] = np.clip(dados[coluna], limite_inferior, limite_superior)
        else:
            # ----------------------------
            # Método IQR
            # ----------------------------
            Q1 = dados[coluna].quantile(0.25)
            Q3 = dados[coluna].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            dados[coluna + '_outlier'] = ((dados[coluna] < limite_inferior) | (dados[coluna] > limite_superior)).astype(int)
            
            dados[coluna] = np.clip(dados[coluna], limite_inferior, limite_superior)
        
    print("\nResumo de outliers detectados (híbrido):")
    print(dados[[coluna for coluna in dados.columns if '_outlier' in coluna]].sum())

    return dados

def limpar_dados(
        dados, 
        colunas_invalidas, 
        limte_inferior_imputacao_simples=0, 
        limite_superior_imputacao_simples=20, 
        limite_inferior_imputacao_avancada=20, 
        limite_superir_imputacao_avancada=40,
        limite_skew=0.5, 
        limite_zscore=3,
        n_vizinhos_knn=5):
    '''
    Aplica os seguintes tratamentos aos dados:
    - Remoção de registros duplicados.
    - Imputação simples (mediana, média ou moda) para ausências entre o limte_inferior_imputacao_simples e limite_superior_imputacao_simples.
    - Imputação avançada (KNNImputer) para ausências entre o limite_inferior_imputacao_avancada e limite_superir_imputacao_avancada.
    - Remoção de colunas cuja ausência exceda o limite_superir_imputacao_avancada.
    - Ajusta os valores de outliers, sinalizando o ajuste em nova coluna.

    Parâmetros:
        dados: conjunto de informações que devem ser tratadas.
        colunas_invalidas: lista de colunas que não podem ter valor zero ou nulo.
        limte_inferior_imputacao_simples: porcentagem mínima de ausências para se aplicar imputação simples.
        limite_superior_imputacao_simples: porcentagem máxima de ausências para se aplicar a imputação simples.
        limite_inferior_imputacao_avancada: porcentagem mínima de ausências para se aplicar a imputação avançada.
        limite_superir_imputacao_avancada: porcentagem máxima de ausências para se aplicar a imputação avançada.
        limite_skew: valor do skew de uma coluna acima do qual: 
            1. se usa mediana ao invés de média na imputação simples; 
            2. se usa IQR ao invés de Z-socre no tratamento de outliers;
        limite_zscore: valor de zscore que define uma coluna como outlier ou não no método Z-score.
        n_vizinhos_knn: quantidade de vizinhos próximos que o KNNImputer deve considerar na imputação avançada

    Retorno:
        dados: conjunto de informações após tratamento.
    '''

    print('Iniciando a limpeza dos dados')

    dados = remover_duplicados(
        dados
    )

    dados = tratar_ausencias(
        dados, 
        colunas_invalidas,
        limte_inferior_imputacao_simples,
        limite_superior_imputacao_simples,
        limite_skew,
        limite_inferior_imputacao_avancada,
        limite_superir_imputacao_avancada,
        n_vizinhos_knn)
    
    dados = tratar_outliers(
        dados,
        limite_skew,
        limite_zscore
    )
    
    # ----------------------------
    # Conferir estatísticas após limpeza
    # ----------------------------
    print('\nResumo estatístico após limpeza:')
    print(dados.describe())

    print('\nFinalizando a limpeza dos dados')

    return dados