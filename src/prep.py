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

# SBERT
modelo_sbert = SentenceTransformer('all-mpnet-base-V2')

def gerar_features_nli(df_subset, nome_conjunto):
    print(f"processando {nome_conjunto}...")
    
    # gera os embeddings apenas da claim e da evidence
    X_c = modelo_sbert.encode(df_subset['claim'].tolist(), show_progress_bar=False)
    X_e = modelo_sbert.encode(df_subset['evidence'].tolist(), show_progress_bar=False)
    
    # subtração absoluta (mede a distancia/contradiçao)
    X_diff = np.abs(X_c - X_e)
    # multiplicaçao (mede a interaçao/sobreposiçao de ideias)
    X_mult = X_c * X_e
    
    # junta tudo numa matriz enxuta (claim, evidence, diferença, multiplicaçao)
    X_denso = np.hstack([X_c, X_e, X_diff, X_mult])
    return X_denso

X_train_dense = gerar_features_nli(df_train, "Treino")
X_val_dense   = gerar_features_nli(df_val, "Validação")
X_test_dense  = gerar_features_nli(df_test, "Teste")

y_train = df_train['label_multiclasse'].values
y_val   = df_val['label_multiclasse'].values
y_test  = df_test['label_multiclasse'].values

# exportacao
print("guardando em 'data/processed'...")
np.save(os.path.join(PASTA_PROCESSED, 'X_train_emb.npy'), X_train_dense)
np.save(os.path.join(PASTA_PROCESSED, 'X_val_emb.npy'), X_val_dense)
np.save(os.path.join(PASTA_PROCESSED, 'X_test_emb.npy'), X_test_dense)

np.save(os.path.join(PASTA_PROCESSED, 'y_train.npy'), y_train)
np.save(os.path.join(PASTA_PROCESSED, 'y_val.npy'), y_val)
np.save(os.path.join(PASTA_PROCESSED, 'y_test.npy'), y_test)

print(f"novo formato: {X_train_dense.shape}.")