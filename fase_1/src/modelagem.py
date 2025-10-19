import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

def avaliar(modelos, X_treino, y_treino, X_teste, y_teste):
    '''
    Verifica qual modelo performa melhor com os dados de treino e teste,
    mostrando os resultados para accuracy_score, recall_score e f1_score.
    '''

    print('\nIniciando a análise dos modelos')

    resultados = []
    
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        y_predicao = modelo.predict(X_teste)

        resultados.append({
            'Nome': nome,
            'Accuracy': accuracy_score(y_teste, y_predicao),
            'Recall': recall_score(y_teste, y_predicao),
            'F1-score': f1_score(y_teste, y_predicao),
            'Modelo': modelo
        })

    df_resultados = pd.DataFrame(resultados)
    print('\nResumo do comparativo\n')
    print(df_resultados.drop(columns=['Modelo']))

    df_resultados_melhor_modelo = df_resultados.loc[df_resultados['Recall'].idxmax()]
    print(f'\nMelhor modelo com base no Recall: {df_resultados_melhor_modelo['Nome']}\n')

    print(df_resultados_melhor_modelo.drop('Modelo').iloc[1:])    

    # melhor_modelo = df_resultados_melhor_modelo['Modelo']
    # y_predito = melhor_modelo.predict(X_teste)
    # print(classification_report(y_teste, y_predito, zero_division=0))        

    print('\nFinalizando a análise dos modelos')

    return df_resultados_melhor_modelo['Modelo']