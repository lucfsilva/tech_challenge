import exploracao
import pre_processamento
import modelagem

def main():
    print('\nInício do Tech Challenge - Fase 1: diagnóstico de diabetes')

    dados = exploracao.carregar('mathchi/diabetes-data-set')
    exploracao.analise_descritiva(dados)
    exploracao.analise_grafica(dados)

    X_treino, X_teste, y_treino, y_teste = pre_processamento.separar(dados, 'Outcome')
    X_treino, X_teste = pre_processamento.limpar(X_treino, X_teste)
    X_treino_padronizado, X_teste_padronizado = pre_processamento.padronizar(X_treino, X_teste)

    modelos_treinados = modelagem.treinar(modelagem.criar_modelos(), X_treino_padronizado, y_treino)
    melhor_modelo = modelagem.avaliar(modelos_treinados, X_teste_padronizado, y_teste)
    modelagem.analisar(melhor_modelo, X_treino_padronizado)

    print('\nFim do Tech Challenge - Fase 1')

if __name__ == "__main__":
    main()