import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve, auc

def analisar_modelos(modelos, X_treino_balanceado, y_treino_balanceado, X_teste_escalonado, y_teste):
    resultados = []

    for nome, modelo in modelos.items():
        modelo.fit(X_treino_balanceado, y_treino_balanceado)
        y_pred = modelo.predict(X_teste_escalonado)
        y_proba = modelo.predict_proba(X_teste_escalonado)[:, 1]

        resultados.append({
            "Nome": nome,
            "Accuracy": accuracy_score(y_teste, y_pred),
            "Precision": precision_score(y_teste, y_pred),
            "Recall": recall_score(y_teste, y_pred),
            "F1-score": f1_score(y_teste, y_pred),
            "AUC": roc_auc_score(y_teste, y_proba),
            "Modelo": modelo
        })

        print(f"\n===== {nome} =====")
        print(classification_report(y_teste, y_pred, digits=3))

    df_resultados = pd.DataFrame(resultados)
    print("\nResumo comparativo dos modelos:")
    print(df_resultados.drop(columns=['Modelo']))

    # plt.figure(figsize=(10, 8))

    # for nome, modelo in modelos.items():
    #     # Gerar probabilidades da classe positiva
    #     y_proba = modelo.predict_proba(X_teste_escalonado)[:, 1]
        
    #     # Calcular ROC
    #     fpr, tpr, thresholds = roc_curve(y_teste, y_proba)
    #     roc_auc = auc(fpr, tpr)
        
    #     # Plotar
    #     plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {roc_auc:.3f})')

    # # Curva do classificador aleatório
    # plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Taxa de erros')
    # plt.ylabel('Taxa de acertos')
    # plt.title('Curva ROC - Comparação de Modelos')
    # plt.legend(loc='lower right')
    # plt.grid(alpha=0.3)
    # plt.show()

    # Foi adotado o Recall como avaliador do melhor modelo pois ele é o mais relacionado a "Verdadeiros positivos", o que é uma métrica importante para diagnósticos médicos.
    # F1-score foi descartado por ser uma média (ponderada) que, apesar de relevante, parece menos importante que o Recall.
    # O Accuracy foi descartado pois os dados estão muito desbalanceados e isso pode afetá-lo negativamente. 
    df_resultados_melhor_modelo = df_resultados.loc[df_resultados['Recall'].idxmax()]
    print('\nMelhor modelo com base no Recall:')
    print(df_resultados_melhor_modelo.drop('Modelo'))    

    return df_resultados_melhor_modelo['Modelo']