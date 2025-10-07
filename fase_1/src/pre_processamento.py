import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.stats import zscore

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
        sufixo_outlier: sufixo usado no nome das novas colunas criadas para identificar outliers
    '''

    print('\nIniciando o tratamento de outliers')
    sufixo_outlier = '_outlier'

    colunas_numericas = dados.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for coluna in colunas_numericas:
        skew_value = dados[coluna].skew()
        
        if abs(skew_value) <= limite_skew:
            # ----------------------------
            # Método Z-score
            # ----------------------------
            z_scores = zscore(dados[coluna], nan_policy='omit')
            dados[coluna + sufixo_outlier] = (np.abs(z_scores) > limite_zscore).astype(int)
            
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
            
            dados[coluna + sufixo_outlier] = ((dados[coluna] < limite_inferior) | (dados[coluna] > limite_superior)).astype(int)
            
            dados[coluna] = np.clip(dados[coluna], limite_inferior, limite_superior)
        
    print("\nResumo de outliers detectados (híbrido):")
    print(dados[[coluna for coluna in dados.columns if sufixo_outlier in coluna]].sum())

    print('\nFinalizando o tratamento de outliers')

    return dados, sufixo_outlier

def escalonar(escalonador, X_treino, X_teste):
    '''
    Escalona o conteúdo das colunas

    Parâmetros:
        scaler: escalonador que será usado
        X_treino: dados de treino
        X_teste: dados de teste

    Retorno:
        X_treino após escalonamento
        X_teste após escalonamento
    '''

    print('\nIniciando o escalonamento')
    
    X_treino_escalonado = escalonador.fit_transform(X_treino)
    X_teste_escalonado = escalonador.transform(X_teste)
    
    X_treino_escalonado_df = pd.DataFrame(X_treino_escalonado, columns=X_treino.columns)
    X_teste_escalonado_df = pd.DataFrame(X_teste_escalonado, columns=X_teste.columns)

    print('\nFinalizando o escalonamento')

    return X_treino_escalonado_df, X_teste_escalonado_df

def balancear(X_treino, y_treino):
    '''
    Gera novos registros para equilibrar a quantidade deles de acordo com a coluna_target

    Parâmetros:
        dados: conjunto de informações que devem ser equilibradas.
        coluna_target (str): nome da coluna que identifica se um registro é verdadeiro ou falso para a pergunta que se quer responder. 
            Exemplo: na análise de dados médicos, a coluna_target pode ser aquela que mostra um diagnóstico como positivo ou negativo.

    Retorno:
        dados: conjunto de informações após equilibrio.
    '''

    print('\nIniciando o escalonamento')

    print("Distribuição original das classes:")
    print(Counter(y_treino))

    smote = SMOTE(random_state=42)
    X_treino_balanceado, y_treino_balanceado = smote.fit_resample(X_treino, y_treino)

    print("\nDistribuição após SMOTE:")
    print(Counter(y_treino_balanceado))

    X_treino_balanceado_df = pd.DataFrame(X_treino_balanceado, columns=X_treino.columns)    
    y_treino_balanceado_df = pd.Series(y_treino_balanceado, columns=y_treino.columns)    

    print('\nFinalizando o escalonamento')

    return X_treino_balanceado_df, y_treino_balanceado_df