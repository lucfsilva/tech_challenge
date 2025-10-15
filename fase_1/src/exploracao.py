import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, shapiro

def analisar_dados(dados, coluna_target, metodo_correlacao='pearson'):
    '''
    Mostra informações relevantes sobre os dados para que se possa tomar decisões 
    quanto a limpeza deles antes de usá-los no treinamento de uma IA.

    Parâmetros:
        dados: conjunto de informações que devem ser analisadas.
        coluna_target (str): nome da coluna que identifica se um registro é verdadeiro ou falso para a pergunta que se quer responder. 
            Exemplo: na análise de dados médicos, a coluna_target pode ser aquela que mostra um diagnóstico como positivo ou negativo.
    '''

    print('Iniciando análise dos dados')

    # ----------------------------
    # Cópia dos dados, porém sem a coluna_target
    # ----------------------------
    dados_sem_coluna_target = dados.drop(columns=[coluna_target])

    # ----------------------------
    # Estatísticas descritivas
    # ----------------------------
    print('\nDimensão do dataset:', dados.shape)

    print('\nEstrutura do dataset:')
    print(dados.info())

    print('\nPrimeiras linhas do dataset:')
    print(dados.head())

    print('\nResumo estatístico:')
    print(dados.describe())

    print(f'\nContagem de valores por classe em {coluna_target}:')
    print(dados[coluna_target].value_counts())

    # ----------------------------
    # Visualização da variável coluna_target
    # ----------------------------
    plt.figure(figsize=(6,4))
    sns.countplot(x=coluna_target, data=dados, palette='Set2', hue=coluna_target, legend=False)
    plt.title(f'Distribuição da variável ({coluna_target})')
    plt.xlabel(f'{coluna_target} (0 = Não diabético, 1 = Diabético)')
    plt.ylabel('Contagem')
    plt.show()

    # ----------------------------
    # Histogramas das variáveis numéricas
    # ----------------------------
    dados_sem_coluna_target.hist(bins=20, figsize=(14,10), edgecolor='black')
    plt.suptitle('Distribuição das variáveis numéricas', fontsize=16)
    plt.show()

    # ----------------------------
    # Boxplots das variáveis numéricas, comparando-as com o coluna_target
    # ----------------------------
    plt.figure(figsize=(14,10))
    for i, coluna in enumerate(dados_sem_coluna_target.columns):
        plt.subplot(3, 3, i+1)
        sns.boxplot(y=coluna, x=coluna_target, data=dados, palette='Set2', hue=coluna_target, legend=False)
        plt.title(f'{coluna} vs {coluna_target}')
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # KDE (distribuição por classe)
    # ----------------------------
    plt.figure(figsize=(14,10))
    for i, coluna in enumerate(dados_sem_coluna_target.columns):
        plt.subplot(3, 3, i+1)
        sns.kdeplot(data=dados, x=coluna, hue=coluna_target, fill=True, common_norm=False, alpha=0.5, palette="Set2")
        plt.title(f'Distribuição de {coluna} por {coluna_target}')
    plt.tight_layout()
    plt.show()    

    # ----------------------------
    # Correlação entre variáveis
    # ----------------------------

    print("Matriz de correlação:")
    matriz_de_correlacao = dados.corr(method=metodo_correlacao)
    print(matriz_de_correlacao)

    plt.figure(figsize=(10,8))
    sns.heatmap(matriz_de_correlacao, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de calor da matriz de correlação')
    plt.show()

    correlacao = matriz_de_correlacao[coluna_target].drop(coluna_target)
    ranking = correlacao.abs().sort_values(ascending=False)

    print("\nRanking das variáveis mais correlacionadas com Outcome:")
    print(ranking)

    # ----------------------------
    # Testes estatísticos
    # ----------------------------
    print("\n===== Testes Estatísticos =====")
    grupo0 = dados[dados[coluna_target] == 0]
    grupo1 = dados[dados[coluna_target] == 1]

    for coluna in dados_sem_coluna_target.columns:
        print(f"\n--- {coluna} ---")
        
        # Teste de normalidade (Shapiro-Wilk)
        stat0, p0 = shapiro(grupo0[coluna])
        stat1, p1 = shapiro(grupo1[coluna])
        
        if p0 > 0.05 and p1 > 0.05:
            # Se ambas distribuições são normais -> teste t
            stat, p = ttest_ind(grupo0[coluna], grupo1[coluna])
            teste = "t de Student (paramétrico)"
        else:
            # Se não são normais -> Mann-Whitney
            stat, p = mannwhitneyu(grupo0[coluna], grupo1[coluna])
            teste = "Mann-Whitney U (não-paramétrico)"
        
        print(f"Teste: {teste}")
        print(f"Estatística = {stat:.4f}, p-valor = {p:.4f}")
        if p < 0.05:
            print("👉 Diferença significativa entre os grupos")
        else:
            print("👉 Não há diferença significativa")    

    print('\nFinalizando análise dos dados')