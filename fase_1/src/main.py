import exploracao
import pre_processamento
import modelagem

def main():
    '''
    Início da aplicação.
    '''

    print('\nInício do Tech Challenge - Fase 1: diagnóstico de diabetes')

    dados = exploracao.carregar('mathchi/diabetes-data-set')
    exploracao.analise_descritiva(dados)
    exploracao.analise_grafica(dados)

    X_treino, X_teste, y_treino, y_teste = pre_processamento.separar(dados, 'Outcome')
    X_treino_imputado, X_teste_imputado = pre_processamento.imputar(X_treino, X_teste)
    # X_treino_balanceado, y_treino_balanceado = pre_processamento.balancear(X_treino_imputado, y_treino)
    # X_treino_padronizado, X_teste_padronizado = pre_processamento.padronizar(X_treino_balanceado, X_teste_imputado)
    X_treino_padronizado, X_teste_padronizado = pre_processamento.padronizar(X_treino_imputado, X_teste_imputado)
    
    # modelos_treinados = modelagem.treinar(modelagem.criar_modelos(), X_treino_padronizado, y_treino_balanceado)
    modelos_treinados = modelagem.treinar(modelagem.criar_modelos(), X_treino_padronizado, y_treino)
    melhor_modelo = modelagem.avaliar(modelos_treinados, X_teste_padronizado, y_teste)
    modelagem.analisar(melhor_modelo, X_treino_padronizado)

    print('\nFim do Tech Challenge - Fase 1')

if __name__ == "__main__":
    main()