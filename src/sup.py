import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier

# config
DIRETORIO_SRC = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BASE = os.path.dirname(DIRETORIO_SRC)
PASTA_PROCESSED = os.path.join(DIRETORIO_BASE, "data", "processed")
PASTA_GRAFICOS = os.path.join(DIRETORIO_BASE, "resultados" , "sup" , "graficos")
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

# carregamento dados
X_train = np.load(os.path.join(PASTA_PROCESSED, 'X_train_emb.npy'), allow_pickle=True)
X_val   = np.load(os.path.join(PASTA_PROCESSED, 'X_val_emb.npy'), allow_pickle=True)
X_test  = np.load(os.path.join(PASTA_PROCESSED, 'X_test_emb.npy'), allow_pickle=True)

y_train = np.load(os.path.join(PASTA_PROCESSED, 'y_train.npy'), allow_pickle=True)
y_val   = np.load(os.path.join(PASTA_PROCESSED, 'y_val.npy'), allow_pickle=True)
y_test  = np.load(os.path.join(PASTA_PROCESSED, 'y_test.npy'), allow_pickle=True)

print(f"tamanho do treino antes do SMOTE: {X_train.shape}")

# aplicar SMOTE no conjunto de treino
print("gerando dados sintéticos para balancear as classes...")
smote = SMOTE(random_state=42)
X_train_balanceado, y_train_balanceado = smote.fit_resample(X_train, y_train)

print(f"tamanho do treino depois do SMOTE: {X_train_balanceado.shape}")

# func visualizacao
def plotar_matriz_confusao(y_true, y_pred, nome_modelo):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0 (Suporta)', '1 (Neutra)', '2 (Refuta)'], 
                yticklabels=['0 (Suporta)', '1 (Neutra)', '2 (Refuta)'])
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.ylabel('Gabarito (Real)')
    plt.xlabel('Previsão do Modelo')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, f'matriz_{nome_modelo}.png'))
    plt.close()

def plotar_curva_aprendizado(estimator, X, y, nome_modelo):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=3, scoring='f1_macro', n_jobs=-1, 
        train_sizes=np.linspace(0.25, 1.0, 4)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, 'o-', color="navy", label="F1 Treino")
    plt.plot(train_sizes, val_mean, 'o-', color="c", label="F1 Validação")
    plt.title(f'Curva de Aprendizado - {nome_modelo}')
    plt.xlabel('Volume de Treinamento')
    plt.ylabel('F1 Macro')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, f'curva_{nome_modelo}.png'))
    plt.close()


modelos = {
    "Regressao_Logistica": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    "SVM_Linear": LinearSVC(max_iter=3000, class_weight='balanced', random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1),
    "Rede_Neural_MLP": MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', alpha=0.01, learning_rate_init=0.001, max_iter=500, random_state=42, early_stopping=True),
    "XGBoost": XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=42, n_jobs=-1, tree_method='hist')
}

# subsample = usa apenas 80% das linhas em cada arvore (adc aleatoriedade)
# colsample_bytree = usa apenas 80% das colunas em cada arvore
# reg_lambda = pune o modelo se ele tentar decorar

# treino, avaliacao e graficos
for nome, modelo in modelos.items():
    print(f"\n[{nome}] treinando o modelo...")
    modelo.fit(X_train_balanceado, y_train_balanceado)

    y_pred_val = modelo.predict(X_val)
    print(f"[{nome}] resultados na validacao:")
    print(classification_report(y_val, y_pred_val, zero_division=0))
    
    print(f"[{nome}] gerando matriz de confusao...")
    plotar_matriz_confusao(y_val, y_pred_val, nome)
    
    print(f"[{nome}] gerando curva de aprendizado...")
    plotar_curva_aprendizado(modelo, X_train, y_train, nome)

print("\ngráficos gerados")