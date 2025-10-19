import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- 1. Carregar os dados ---
dados = pd.read_csv('pasta\\arquivo.csv') # colocar o endereço real do arquivo

# --- 2. Substituir zeros inválidos por NaN ---
cols_invalidas = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
dados[cols_invalidas] = dados[cols_invalidas].replace(0, np.nan)

# --- 3. Separar features e alvo ---
X = dados.drop('Outcome', axis=1)
y = dados['Outcome']

# --- 4. Separar treino e teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Criar pipeline de pré-processamento ---
# Etapas: imputação + padronização
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, X.columns)
])

# --- 6. Criar pipeline completo com balanceamento (SMOTE) ---
# Você pode plugar aqui qualquer modelo no final
from sklearn.linear_model import LogisticRegression

pipeline = ImbPipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', LogisticRegression(max_iter=1000, class_weight=None))
])

# --- 7. Treinar o modelo ---
pipeline.fit(X_train, y_train)

# --- 8. Avaliar ---
from sklearn.metrics import classification_report, confusion_matrix

y_pred = pipeline.predict(X_test)

print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))