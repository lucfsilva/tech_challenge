def mostrar_informacoes_relevantes(dados):
    """
    Exibe detalhes importantes sobre os dados para a tomada de decisões sobre como trata-los para o treinamento de uma IA
    
    Args:
        dados (DadaFrame): conjunto de informações que será avaliado.
    Returns:
        dados (DataFrame): retorna os mesmos dados recebidos por parâmetro
    """

    print('Quantidade de linhas e colunas dos dados:')
    print(dados.shape)
    print('\n')
    
    print('Informações sobre a estrutura dos dados:')
    print(dados.info())
    print('\n')
    
    print('Estatísticas básicas sobre o conteúdo dos dados:')
    print(dados.describe())
    print('\n')

    print('Primeiros registros dos dados:')
    print(dados.head())

    return dados