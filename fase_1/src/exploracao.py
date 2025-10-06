import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, shapiro

def analisar_estatisticas_descritivas(dados, target):

    # ----------------------------
    # Estatísticas descritivas
    # ----------------------------
    print('Dimensão do dataset:', dados.shape)

    print('\nEstrutura do dataset:')
    print(dados.info())

    print('\nPrimeiras linhas do dataset:')
    print(dados.head())

    print('\nResumo estatístico:')
    print(dados.describe())

    print(f'\nContagem de valores por classe em {target}:')
    print(dados[target].value_counts())

    # ----------------------------
    # Visualização da variável alvo
    # ----------------------------
    plt.figure(figsize=(6,4))
    sns.countplot(x=target, data=dados, palette='Set2', hue=target, legend=False)
    plt.title(f'Distribuição da variável alvo ({target})')
    plt.xlabel(f'{target} (0 = Não, 1 = Sim)')
    plt.ylabel('Contagem')
    plt.show()

    # ----------------------------
    # Histogramas das variáveis numéricas
    # ----------------------------
    dados.hist(bins=20, figsize=(14,10), edgecolor='black')
    plt.suptitle('Distribuição das variáveis numéricas', fontsize=16)
    plt.show()

    # ----------------------------
    # Boxplots das variáveis numéricas
    # ----------------------------
    plt.figure(figsize=(14,10))
    dados_sem_target = dados.drop(columns=[target])
    for i, coluna in enumerate(dados_sem_target.columns):
        plt.subplot(3, 3, i+1)
        sns.boxplot(y=coluna, x=target, data=dados, palette='Set2', hue=target, legend=False)
        plt.title(f'{coluna} vs {target}')
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
    grupo0 = dados[dados[target] == 0]
    grupo1 = dados[dados[target] == 1]

    dados_sem_target = dados.drop(columns=[target])
    for col in dados_sem_target.columns:
        print(f"\n--- {col} ---")
        
        # Teste de normalidade (Shapiro-Wilk)
        stat0, p0 = shapiro(grupo0[col])
        stat1, p1 = shapiro(grupo1[col])
        
        if p0 > 0.05 and p1 > 0.05:
            # Se ambas distribuições são normais -> teste t
            stat, p = ttest_ind(grupo0[col], grupo1[col])
            teste = "t de Student (paramétrico)"
        else:
            # Se não são normais -> Mann-Whitney
            stat, p = mannwhitneyu(grupo0[col], grupo1[col])
            teste = "Mann-Whitney U (não-paramétrico)"
        
        print(f"Teste: {teste}")
        print(f"Estatística = {stat:.4f}, p-valor = {p:.4f}")
        if p < 0.05:
            print("👉 Diferença significativa entre os grupos")
        else:
            print("👉 Não há diferença significativa")    