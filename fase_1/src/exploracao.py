import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, shapiro

def analisar_dados(dados, alvo):

    print('Iniciando análise dos dados')

    # ----------------------------
    # Cópia dos dados, porém sem a coluna "alvo"
    # ----------------------------
    dados_sem_alvo = dados.drop(columns=[alvo])

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

    print(f'\nContagem de valores por classe em {alvo}:')
    print(dados[alvo].value_counts())

    # ----------------------------
    # Visualização da variável alvo
    # ----------------------------
    plt.figure(figsize=(6,4))
    sns.countplot(x=alvo, data=dados, palette='Set2', hue=alvo, legend=False)
    plt.title(f'Distribuição da variável alvo ({alvo})')
    plt.xlabel(f'{alvo} (0 = Não, 1 = Sim)')
    plt.ylabel('Contagem')
    plt.show()

    # ----------------------------
    # Histogramas das variáveis numéricas
    # ----------------------------
    dados.hist(bins=20, figsize=(14,10), edgecolor='black')
    plt.suptitle('Distribuição das variáveis numéricas', fontsize=16)
    plt.show()

    # ----------------------------
    # Boxplots das variáveis numéricas, comparando-as com o alvo
    # ----------------------------
    plt.figure(figsize=(14,10))
    for i, coluna in enumerate(dados_sem_alvo.columns):
        plt.subplot(3, 3, i+1)
        sns.boxplot(y=coluna, x=alvo, data=dados, palette='Set2', hue=alvo, legend=False)
        plt.title(f'{coluna} vs {alvo}')
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # KDE (distribuição por classe)
    # ----------------------------
    plt.figure(figsize=(14,10))
    for i, coluna in enumerate(dados_sem_alvo.columns):
        plt.subplot(3, 3, i+1)
        sns.kdeplot(data=dados, x=coluna, hue=alvo, fill=True, common_norm=False, alpha=0.5, palette="Set2")
        plt.title(f'Distribuição de {coluna} por {alvo}')
    plt.tight_layout()
    plt.show()    

    # ----------------------------
    # Correlação entre variáveis
    # ----------------------------
    plt.figure(figsize=(10,8))
    sns.heatmap(dados.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlação')
    plt.show()

    # ----------------------------
    # Testes estatísticos
    # ----------------------------
    print("\n===== Testes Estatísticos =====")
    grupo0 = dados[dados[alvo] == 0]
    grupo1 = dados[dados[alvo] == 1]

    for coluna in dados_sem_alvo.columns:
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