import os
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# config
DIRETORIO_SRC = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BASE = os.path.dirname(DIRETORIO_SRC)
PASTA_PROCESSED = os.path.join(DIRETORIO_BASE, "data", "processed")


# carregamento dos dados
X_train = np.load(os.path.join(PASTA_PROCESSED, 'X_train_emb.npy'))
X_val   = np.load(os.path.join(PASTA_PROCESSED, 'X_val_emb.npy'))
X_test  = np.load(os.path.join(PASTA_PROCESSED, 'X_test_emb.npy'))

y_train = np.load(os.path.join(PASTA_PROCESSED, 'y_train.npy'))
y_val   = np.load(os.path.join(PASTA_PROCESSED, 'y_val.npy'))
y_test  = np.load(os.path.join(PASTA_PROCESSED, 'y_test.npy'))

print(f"dados carregados, formato do Treino: {X_train.shape}")

# config modelos
modelos = {    
    "Regressão Logistica": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
}

# treino e validaçao
for nome, modelo in modelos.items():
    print(f"\na treinar {nome}...")
    modelo.fit(X_train, y_train)
    
    y_pred_val = modelo.predict(X_val)
    print(f"resultados na validação: {nome} ---")
    print(classification_report(y_val, y_pred_val))

print("\nconcluído!")