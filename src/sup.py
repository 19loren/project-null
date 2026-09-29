import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import load_npz
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# config
DIRETORIO_SRC = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BASE = os.path.dirname(DIRETORIO_SRC)

PASTA_PROCESSED = os.path.join(DIRETORIO_BASE, "data", "processed")
PASTA_RESULTADOS = os.path.join(DIRETORIO_BASE, "resultados", "sup")

os.makedirs(PASTA_RESULTADOS, exist_ok=True)

print("iniciando treinamento supervisionado")

# carregamento data
try:
    X_train = load_npz(os.path.join(PASTA_PROCESSED, 'X_train_tfidf.npz'))
    X_val   = load_npz(os.path.join(PASTA_PROCESSED, 'X_val_tfidf.npz'))
    X_test  = load_npz(os.path.join(PASTA_PROCESSED, 'X_test_tfidf.npz'))

    y_train_raw = np.load(os.path.join(PASTA_PROCESSED, 'y_train.npy'), allow_pickle=True)
    y_val_raw   = np.load(os.path.join(PASTA_PROCESSED, 'y_val.npy'), allow_pickle=True)
    y_test_raw  = np.load(os.path.join(PASTA_PROCESSED, 'y_test.npy'), allow_pickle=True)
    print(f"matrizes carregadas. treino: {X_train.shape}")
except FileNotFoundError:
    print("ERRO: matrizes não encontradas. por favor, rode o 'prep.py' primeiro")
    exit()

# padronizacao dos rotulos numericos (XGBoost)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_raw)
y_val   = encoder.transform(y_val_raw)
y_test  = encoder.transform(y_test_raw)
classes_nomes = encoder.classes_

# visualizaçao e funçoes auxiliares
AZUL, TURQUESA, TURQUESA_CLARO, FUNDO_CLARO = "#183A8F", "#4EB8C5", "#7FD0D8", "#F1FAFB"
cmap_grupo = LinearSegmentedColormap.from_list("profilaxia", [FUNDO_CLARO, TURQUESA_CLARO, TURQUESA, AZUL])

def nome_arquivo_limpo(nome):
    return nome.replace(" ", "_").replace("-", "").replace("ç", "c").replace("ã", "a").strip()

def calcular_metricas(y_real, y_pred):
    return {
        "Acurácia": accuracy_score(y_real, y_pred),
        "Precisão macro": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "Recall macro": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "F1 macro": f1_score(y_real, y_pred, average="macro", zero_division=0)
    }

def salvar_grafico_metricas(nome, metricas):
    nomes, valores = ["Acurácia", "Precisão macro", "Recall macro", "F1 macro"], list(metricas.values())
    plt.figure(figsize=(7, 4))
    barras = plt.bar(nomes, valores, color=[AZUL, TURQUESA, TURQUESA_CLARO, AZUL])
    plt.ylim(0, 1)
    plt.ylabel("Valor")
    plt.title(f"{nome} - métricas na validação")
    for barra, valor in zip(barras, valores):
        plt.text(barra.get_x() + barra.get_width() / 2, valor + 0.02, f"{valor:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_RESULTADOS, f"metricas_{nome_arquivo_limpo(nome)}.png"), dpi=300)
    plt.close()

def salvar_matriz_confusao(nome, y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(6.5, 5.2))
    plt.imshow(cm, cmap=cmap_grupo)
    plt.title(f"Matriz de confusão - {nome}", color=AZUL)
    plt.xlabel("Classe prevista", color=AZUL)
    plt.ylabel("Classe real", color=AZUL)
    plt.xticks(range(len(classes_nomes)), classes_nomes)
    plt.yticks(range(len(classes_nomes)), classes_nomes)
    max_val = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = cm[i, j]
            cor = "white" if valor > max_val * 0.45 else AZUL
            plt.text(j, i, str(valor), ha="center", va="center", color=cor, fontweight="bold")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_RESULTADOS, f"matriz_{nome_arquivo_limpo(nome)}.png"), dpi=300)
    plt.close()

def salvar_curva_aprendizado(modelo, nome, X_tr, y_tr, X_v, y_v):
    X_tr_shuf, y_tr_shuf = shuffle(X_tr, y_tr, random_state=42)
    tamanhos, f1_treino, f1_val = [], [], []
    total = X_tr_shuf.shape[0]

    for fracao in [0.1, 0.3, 0.5, 0.7, 1.0]:
        qtd = int(total * fracao)
        X_p, y_p = X_tr_shuf[:qtd], y_tr_shuf[:qtd]
        mod_temp = clone(modelo)
        mod_temp.fit(X_p, y_p)
        
        f1_treino.append(f1_score(y_p, mod_temp.predict(X_p), average="macro", zero_division=0))
        f1_val.append(f1_score(y_v, mod_temp.predict(X_v), average="macro", zero_division=0))
        tamanhos.append(qtd)

    plt.figure(figsize=(8, 4.5))
    plt.plot(tamanhos, f1_treino, marker="o", label="F1 Treino", color=AZUL)
    plt.plot(tamanhos, f1_val, marker="o", label="F1 Validação", color=TURQUESA)
    plt.title(f"Curva de aprendizado - {nome}")
    plt.xlabel("Exemplos no treinamento")
    plt.ylabel("F1 Macro")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_RESULTADOS, f"curva_{nome_arquivo_limpo(nome)}.png"), dpi=300)
    plt.close()

# treinamento dos modelos
modelos = {
    "Regressão Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "Arvore de Decisão": DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, class_weight="balanced", random_state=42),
    "SVM Linear": LinearSVC(C=0.15, class_weight="balanced", max_iter=5000)
}

resultados_validacao = {}
modelos_treinados = {}

for nome, modelo in modelos.items():
    print(f"\nA processar: {nome}...")
    if "Baseline" not in nome:
        salvar_curva_aprendizado(modelo, nome, X_train, y_train, X_val, y_val)

    modelo.fit(X_train, y_train)
    modelos_treinados[nome] = modelo
    y_pred_val = modelo.predict(X_val)

    metricas = calcular_metricas(y_val, y_pred_val)
    resultados_validacao[nome] = metricas
    salvar_grafico_metricas(nome, metricas)
    salvar_matriz_confusao(nome, y_val, y_pred_val)

# seleçao e teste final
df_resultados = pd.DataFrame.from_dict(resultados_validacao, orient="index")
candidatos = df_resultados.drop("Baseline - Dummy", errors="ignore")

plt.figure(figsize=(9, 5))
candidatos["F1 macro"].sort_values().plot(kind="barh", color=TURQUESA)
plt.title("Comparativo de Candidatos - F1 Macro na validação")
plt.xlabel("Pontuação")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_RESULTADOS, "comparativo_f1_macro.png"), dpi=300)
plt.close()

modelo_selecionado_nome = "Random Forest"
modelo_selecionado = modelos_treinados[modelo_selecionado_nome]
print(f"\nModelo selecionado: {modelo_selecionado_nome}")

y_pred_test = modelo_selecionado.predict(X_test)
print(f"\nDESEMPENHO NO CONJUNTO DE TESTE")
print(classification_report(y_test, y_pred_test, target_names=[str(c) for c in classes_nomes], zero_division=0))

salvar_matriz_confusao(f"Teste_Final_{modelo_selecionado_nome}", y_test, y_pred_test) 
print(f"\ntodos os gráficos guardados na pasta: {PASTA_RESULTADOS}")