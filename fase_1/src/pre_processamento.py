import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def separar(dados: pd.DataFrame, target: str):
    '''
    Aplica padronização nas colunas, para que todas tenham uma escala semelhante.

    Parâmetros:
        X_treino: dados de treino
        X_teste: dados de teste

    Retorno:
        X_treino escalonado
        X_teste escalonado
    '''
    
    X_dados = dados.drop(columns=[target], axis=1)
    y_dados = dados[target]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X_dados, y_dados, test_size=0.2, random_state=42, stratify=y_dados)

    return X_treino, X_teste, y_treino, y_teste

def imputar(X_treino: pd.DataFrame, X_teste: pd.DataFrame) -> pd.DataFrame:
    '''
    Trata as informações que podem prejudicar o treinamento do modelo, como zeros inválidos.
    Foi considerado que as seguintes colunas devem sempre ter um valor maior do que zero:
    - Glucose
    - BloodPressure
    - SkinThickness
    - Insulin
    - BMI
    - Age

    Parâmetros:
        dados: DataFrame com as informações que devem ser tratadas

    Retorno:
        DataFrame com as informações já tratadas
    '''
    print('\nIniciando a limpeza dos dados')

    colunas = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    X_treino[colunas] = X_treino[colunas].replace(0, np.nan)
    X_teste[colunas] = X_teste[colunas].replace(0, np.nan)

    imputer = KNNImputer(n_neighbors=5)
    X_treino[colunas] = imputer.fit_transform(X_treino[colunas])
    X_teste[colunas] = imputer.transform(X_teste[colunas])

    print('\nFinalizando a limpeza dos dados')

    return X_treino, X_teste

def padronizar(X_treino: pd.DataFrame, X_teste: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Aplica padronização nas colunas, para que todas tenham uma escala semelhante.

    Parâmetros:
        X_treino: dados de treino
        X_teste: dados de teste

    Retorno:
        X_treino escalonado
        X_teste escalonado
    '''

    print('\nIniciando o escalonamento dos dados')
    
    scaler = RobustScaler()
    X_treino_padronizado = scaler.fit_transform(X_treino)
    X_teste_padronizado = scaler.transform(X_teste)
    
    X_treino_padronizado_df = pd.DataFrame(X_treino_padronizado, columns=X_treino.columns)
    X_teste_padronizado_df = pd.DataFrame(X_teste_padronizado, columns=X_teste.columns)

    print('\nFinalizando o escalonamento dos dados')

    return X_treino_padronizado_df, X_teste_padronizado_df

def balancear(X_treino: pd.DataFrame, y_treino: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Equilibra a quantidade de dados por diagóstico

    Parâmetros:
        X_treino: características de treino
        y_treino: diagnósticos de treino

    Retorno:
        X_treino balanceado
        y_treino balanceado
    '''

    print('\nIniciando o balanceamento dos dados')
    X_treino_balanceado, y_treino_balanceado = SMOTE().fit_resample(X_treino, y_treino)
    print('\nFinalizando o balanceamento dos dados')

    return X_treino_balanceado, y_treino_balanceado    