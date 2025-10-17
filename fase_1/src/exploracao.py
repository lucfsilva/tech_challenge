import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analise_descritiva(dados: pd.DataFrame):
    '''
    Descreve características relevantes dos dados, como:
    - Dimensão (quantidade de linhas e colunas)
    - Estrutura (tipo de informação, quantidade de nulos, etc)
    - Primeiras linhas
    - Resumo estatístico por coluna (quantidade, média, mínimo, máximo, etc)

    Parâmetros:
        dados: DataFrame que deve ser exibido
    '''

    print('Iniciando análise descritiva dos dados')

    print('\nDimensão:', dados.shape)

    print('\nEstrutura:')
    print(dados.info())

    print('\nPrimeiras linhas:')
    print(dados.head())

    print('\nResumo estatístico:')
    print(dados.describe())

    print('\nFinalizando análise descritiva dos dados')

def analise_grafica(dados: pd.DataFrame):
    '''
    Mostra gráficos com características relevantes dos dados, como:
    - Distribuição do diagnóstico
    - Boxplot das características clínicas.
    - Proporção de outliers por característica clínica.
    - Mapa de calor entre características clínicas e diagnóstico.
    - Ranking de correlação entre características clínicas e diagnóstico.

    Parâmetros:
        dados: DataFrame que deve ser analisado
    '''

    dados = dados.rename(columns={
        'Pregnancies': 'Gestações',
        'Glucose': 'Glicose',
        'BloodPressure': 'Pressão arterial',
        'SkinThickness': 'Espessura da pele',
        'Insulin': 'Insulina',
        'BMI': 'IMC',
        'DiabetesPedigreeFunction': 'Hereditariedade',
        'Age': 'Idade',
        'Outcome': 'Diagnóstico'
    })

    # ----------------------------
    # Distribuição do diagnóstico
    # ----------------------------
    contagens = dados['Diagnóstico'].value_counts().sort_index()

    plt.figure(figsize=(5, 5))
    plt.pie(
        contagens,
        labels=['Não diabético', 'Diabético'],
        autopct='%1.2f%%',
        startangle=90,
        colors=sns.color_palette('Set2')
    )
    plt.title('Distribuição de diagnósticos')
    plt.show()

    # ----------------------------
    # Boxplot das características clínicas
    # ----------------------------
    dados_sem_diagnostico = dados.drop('Diagnóstico', axis=1)

    plt.figure(figsize=(10,8))
    plt.suptitle('Características clínicas', fontsize=14)
    for i, coluna in enumerate(dados_sem_diagnostico.columns):
        plt.subplot(3, 3, i+1)
        sns.boxplot(y=coluna, data=dados_sem_diagnostico, color='skyblue')
        plt.title(f'{coluna}')
        plt.ylabel('')
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Proporção de outliers por característica clínica, usando IQR
    # ----------------------------
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    axes = axes.flatten()

    for i, coluna in enumerate(dados_sem_diagnostico.columns):
        Q1 = dados_sem_diagnostico[coluna].quantile(0.25)
        Q3 = dados_sem_diagnostico[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        outliers = ((dados_sem_diagnostico[coluna] < limite_inferior) | 
                    (dados_sem_diagnostico[coluna] > limite_superior))

        num_outliers = outliers.sum()
        num_normais = len(outliers) - num_outliers

        eixos = axes[i]
        eixos.pie(
            [num_normais, num_outliers],
            labels=['Normais', 'Outliers'],
            autopct='%1.1f%%',
            colors=['skyblue', 'salmon'],
            startangle=90
        )
        eixos.set_title(coluna)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Proporção de outliers por característica clínica', fontsize=14)
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Mapa de calor entre características clínicas e diagnóstico
    # ----------------------------
    matriz_de_correlacao = dados.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(matriz_de_correlacao, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de calor')
    plt.show()

    # ----------------------------
    # Ranking da correlação entre características clínicas e diagnóstico
    # ----------------------------
    correlacao = matriz_de_correlacao['Diagnóstico'].drop('Diagnóstico')
    coluna_X = 'Característica clínica'
    coluna_y = 'Correlação com diagnóstico'
    ranking = correlacao.abs().sort_values(ascending=False)
    ranking_df = ranking.reset_index()
    ranking_df.columns = [coluna_X, coluna_y]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=ranking_df, x=coluna_X, y=coluna_y, color='skyblue')
    plt.title('Ranking de correlação do diagnóstico', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()