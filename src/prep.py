import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer

# config
DIRETORIO_SRC = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BASE = os.path.dirname(DIRETORIO_SRC)
PASTA_RAW = os.path.join(DIRETORIO_BASE, "data", "raw")
PASTA_PROCESSED = os.path.join(DIRETORIO_BASE, "data", "processed")

# carregamento dos dados
caminho_excel = os.path.join(PASTA_RAW, 'healthver_preparado_completo_3classes.xlsx')

try:
    df = pd.read_excel(caminho_excel)
    print(f"base carregada: {df.shape[0]} linhas.")
except FileNotFoundError:
    print(f"ERRO: arquivo não encontrado em {caminho_excel}")
    exit()

df['question'] = df['question'].fillna('').astype(str)
df['claim']    = df['claim'].fillna('').astype(str)
df['evidence'] = df['evidence'].fillna('').astype(str)

# divisao (treino / val / teste)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=df['claim']))

df_train = df.iloc[train_idx]
df_temp = df.iloc[temp_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['claim']))

df_val = df_temp.iloc[val_idx]
df_test = df_temp.iloc[test_idx]

print(f"divisao segura: treino ({len(df_train)}), val ({len(df_val)}), teste ({len(df_test)})")

# geracao de embeddings SBERT
modelo_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# gera embeddings Treino
# o SBERT converte cada lista de textos numa matriz densa de 384 colunas numericas
X_tr_q = modelo_sbert.encode(df_train['question'].tolist(), show_progress_bar=True)
X_tr_c = modelo_sbert.encode(df_train['claim'].tolist(), show_progress_bar=False)
X_tr_e = modelo_sbert.encode(df_train['evidence'].tolist(), show_progress_bar=False)
X_train_dense = np.hstack([X_tr_q, X_tr_c, X_tr_e])

# gera embeddings Validaçao
X_val_q = modelo_sbert.encode(df_val['question'].tolist(), show_progress_bar=False)
X_val_c = modelo_sbert.encode(df_val['claim'].tolist(), show_progress_bar=False)
X_val_e = modelo_sbert.encode(df_val['evidence'].tolist(), show_progress_bar=False)
X_val_dense = np.hstack([X_val_q, X_val_c, X_val_e])

# gera embeddings Teste
X_test_q = modelo_sbert.encode(df_test['question'].tolist(), show_progress_bar=False)
X_test_c = modelo_sbert.encode(df_test['claim'].tolist(), show_progress_bar=False)
X_test_e = modelo_sbert.encode(df_test['evidence'].tolist(), show_progress_bar=False)
X_test_dense = np.hstack([X_test_q, X_test_c, X_test_e])

y_train = df_train['label_multiclasse'].values
y_val   = df_val['label_multiclasse'].values
y_test  = df_test['label_multiclasse'].values

# exportaçao (denso)
print("guardando em 'data/processed'...")

# salvamos como numpy array padrao (.npy)
np.save(os.path.join(PASTA_PROCESSED, 'X_train_emb.npy'), X_train_dense)
np.save(os.path.join(PASTA_PROCESSED, 'X_val_emb.npy'), X_val_dense)
np.save(os.path.join(PASTA_PROCESSED, 'X_test_emb.npy'), X_test_dense)

np.save(os.path.join(PASTA_PROCESSED, 'y_train.npy'), y_train)
np.save(os.path.join(PASTA_PROCESSED, 'y_val.npy'), y_val)
np.save(os.path.join(PASTA_PROCESSED, 'y_test.npy'), y_test)

print(f"novo formato denso: {X_train_dense.shape}.")