import traducao
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def treinar(modelos, X_treino, y_treino):
    '''
    Treina uma lista de modelos para posterior avaliação.

    Parâmetros:
        modelos: lista de instâncias dos modelos que serão treinados.
        X_treino: características clínicas que devem ser usadas no treino
        y_treino: diagnósticos que devem ser usados no treino.
    Retorno:
        lista de modelos treinados
    '''

    print('\nIniciando o treino dos modelos')

    resultados = []
    
    for modelo in modelos:
        modelo.fit(X_treino, y_treino)
        resultados.append(modelo)

    print('\nFinalizando o treino dos modelos')

    return resultados

def avaliar(modelos, X_teste, y_teste):
    '''
    Avalia uma lista de modelos, retornando o de melhor performance

    Parâmetros:
        modelos: lista de instâncias dos modelos que serão treinados.
        X_teste: características clínicas que devem ser usadas no teste
        y_teste: diagnósticos que devem ser usados no teste.
    Retorno:
        Modelo de melhor performance.
    '''

    print('\nIniciando a avaliação dos modelos')

    resultados = []

    for modelo in modelos:
        y_previsto = modelo.predict(X_teste)
        nome = type(modelo).__name__

        resultados.append({
            'Modelo': nome,
            'Accuracy': accuracy_score(y_teste, y_previsto),
            'Recall': recall_score(y_teste, y_previsto),
            'F1-score': f1_score(y_teste, y_previsto),
        })

        print(f'\nClassificação do modelo {nome}')
        print(classification_report(y_teste, y_previsto, zero_division=0))        

    df_resultados = pd.DataFrame(resultados)
    print('\nResumo das classificações\n')
    print(df_resultados)

    df_resultados_melhor_modelo = df_resultados.loc[df_resultados['Recall'].idxmax()]
    nome_melhor_modelo = df_resultados_melhor_modelo['Modelo']
    melhor_modelo = next((modelo for modelo in modelos if type(modelo).__name__ == nome_melhor_modelo), None)

    print(f'\nMelhor modelo: {nome_melhor_modelo}\n')
    print(df_resultados_melhor_modelo.drop('Modelo'))

    print('\nFinalizando a avaliação dos modelos')

    return melhor_modelo

def analisar(modelo, X_treino):
    '''
    Analisa quais características clínicas tiveram maior importância nos testes do modelo.

    Parâmetros:
        modelo: modelo de melhor performance
        X_treino: características clínicas usadas no treino do modelo.
    '''

    print('\nIniciando avaliação do modelo')

    X_treino_traduzido = traducao.traduzir(X_treino)

    # Analisando por "feature importance"
    importances = modelo.feature_importances_
    pd.Series(importances, index=X_treino_traduzido.columns).sort_values().plot(kind='barh')
    plt.title('Importância das variáveis')
    plt.tight_layout()
    plt.show()

    # Analisando por "SHAP"
    explainer = shap.Explainer(modelo, X_treino_traduzido)
    shap_values = explainer(X_treino_traduzido)

    shap.summary_plot(shap_values[..., 1], X_treino_traduzido, plot_type='bar')
    # shap.summary_plot(shap_values[..., 1], X_treino_traduzido)

    print('\nFinalizando avaliação do modelo')