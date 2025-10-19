def traduzir(dados):
    '''
    Traduz os nomes das colunas para facilitar a interpretação dos dados.

    Parâmetros:
        dados: DataFrame com os dados cujas colunas devem ser traduzidas
    '''

    dados = dados.rename(columns={
        'Pregnancies': 'Gestações',
        'Glucose': 'Glicose',
        'BloodPressure': 'Pressão',
        'SkinThickness': 'Espessura da pele',
        'Insulin': 'Insulina',
        'BMI': 'IMC',
        'DiabetesPedigreeFunction': 'Hereditariedade',
        'Age': 'Idade',
        'Outcome': 'Diagnóstico'
    })

    return dados