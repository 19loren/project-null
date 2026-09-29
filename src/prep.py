import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

# config
DIRETORIO_SRC = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BASE = os.path.dirname(DIRETORIO_SRC)

PASTA_RAW = os.path.join(DIRETORIO_BASE, "data", "raw")
PASTA_PROCESSED = os.path.join(DIRETORIO_BASE, "data", "processed")

print("iniciando preparação de dados")

# carregamento raw data
caminho_excel = os.path.join(PASTA_RAW, 'healthver_preparado_completo_3classes.xlsx')

try:
    df = pd.read_excel(caminho_excel)
    print(f"base carregada com sucesso: {df.shape[0]} linhas")
except FileNotFoundError:
    print(f"ERRO: arquivo não encontrado em {caminho_excel}")
    print("certifique-se de que moveu o excel para a pasta 'data/raw'")
    exit()

# garantir tratamento de textos nulos e converter para string
df['question'] = df['question'].fillna('').astype(str)
df['claim']    = df['claim'].fillna('').astype(str)
df['evidence'] = df['evidence'].fillna('').astype(str)

# divisao (train / val / test)
# passo 1: separa 70% das claims para o treino e 30% para um bloco temporario
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=df['claim']))

df_train = df.iloc[train_idx]
df_temp = df.iloc[temp_idx]

# passo 2: divide o bloco temporario ao meio (15% validaçao / 15% teste)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp['claim']))

df_val = df_temp.iloc[val_idx]
df_test = df_temp.iloc[test_idx]

print(
    f"divisão segura concluída: treino ({len(df_train)} linhas), "
    f"validação ({len(df_val)} linhas), teste ({len(df_test)} linhas)"
)

# vetorizaçao
print("aplicando vetorização TF-IDF...")
vec_question = TfidfVectorizer(max_features=400,  min_df=2, stop_words='english', ngram_range=(1, 2))
vec_claim    = TfidfVectorizer(max_features=800,  min_df=2, stop_words='english', ngram_range=(1, 2))
vec_evidence = TfidfVectorizer(max_features=1200, min_df=2, stop_words='english', ngram_range=(1, 2))

# ajuste e transformaçao no treino
X_tr_q = vec_question.fit_transform(df_train['question'])
X_tr_c = vec_claim.fit_transform(df_train['claim'])
X_tr_e = vec_evidence.fit_transform(df_train['evidence'])
X_train_stacked = hstack([X_tr_q, X_tr_c, X_tr_e]).tocsr()

# apenas transformaçao na validaçao
X_val_q = vec_question.transform(df_val['question'])
X_val_c = vec_claim.transform(df_val['claim'])
X_val_e = vec_evidence.transform(df_val['evidence'])
X_val_stacked = hstack([X_val_q, X_val_c, X_val_e]).tocsr()

# apenas transformaçao no teste
X_test_q = vec_question.transform(df_test['question'])
X_test_c = vec_claim.transform(df_test['claim'])
X_test_e = vec_evidence.transform(df_test['evidence'])
X_test_stacked = hstack([X_test_q, X_test_c, X_test_e]).tocsr()

y_train = df_train['label_multiclasse'].values
y_val   = df_val['label_multiclasse'].values
y_test  = df_test['label_multiclasse'].values

# exporta pra processed
print("salvando matrizes na pasta 'dados/processed'...")

save_npz(os.path.join(PASTA_PROCESSED, 'X_train_tfidf.npz'), X_train_stacked)
save_npz(os.path.join(PASTA_PROCESSED, 'X_val_tfidf.npz'), X_val_stacked)
save_npz(os.path.join(PASTA_PROCESSED, 'X_test_tfidf.npz'), X_test_stacked)

np.save(os.path.join(PASTA_PROCESSED, 'y_train.npy'), y_train)
np.save(os.path.join(PASTA_PROCESSED, 'y_val.npy'), y_val)
np.save(os.path.join(PASTA_PROCESSED, 'y_test.npy'), y_test)

print(f"o formato final das matrizes é {X_train_stacked.shape}")
print("os dados estao prontos para os modelos")