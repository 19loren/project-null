import os
import joblib
import numpy as np
import requests
import functools
import time
import warnings
from transformers import logging

# Calar os avisos vermelhos
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer

NCBI_API_KEY = "0ee2148580f3394873b20961bddd4488b608"

original_get = requests.get

# interpcetor
@functools.wraps(original_get)
def get_com_chave(*args, **kwargs):
    params = kwargs.get('params', {})
    if params is None:
        params = {}
    
    # injeta a chave em qualquer pedido que o buscador
    if isinstance(params, dict):
        params['api_key'] = NCBI_API_KEY
        kwargs['params'] = params
        
    # adiciona uma micropausa de segurança exigida pelas regras da NCBI
    time.sleep(0.3)
    
    return original_get(*args, **kwargs)

requests.get = get_com_chave

# importa buscador
from buscador import coletar_evidencias, traduzir_Marian

print("Carregando modelo...")
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BASE = os.path.dirname(DIRETORIO_ATUAL)
caminho_modelo = os.path.join(DIRETORIO_BASE, "data", "processed", "modelo_mlp_profilaxia.pkl")

modelo_mlp = joblib.load(caminho_modelo)
motor_semantico = SentenceTransformer('all-mpnet-base-v2')
MAPA_LABELS = {0: "SUPPORTED", 1: "REFUTED", 2: "NEUTRAL"}

def analisar_alegacao(pergunta_pt):
    print(f"\n[1] Traduzindo '{pergunta_pt}'...")
    pergunta_en = traduzir_Marian(pergunta_pt)
    print(f"Tradução: {pergunta_en}")
    
    print(f"[2] Vasculhando o PubMed...")
    
    resultado = coletar_evidencias(pergunta_en, n=20)
    
    if resultado is None or not resultado.get("evidencias"):
        return {"status": "erro", "mensagem": "Não foram encontradas evidências suficientes. A gramática pode ter falhado ou não há artigos Open Access."}

    print(f"[3] A Rede Neural vai classificar as {len(resultado['evidencias'])} evidências válidas...")
    
    vetor_claim = motor_semantico.encode(resultado["claim"])
    evidencias_classificadas = []
    
    for ev in resultado["evidencias"]:
        # limite de leitura de 350 caracteres para evitar overfitting de ruído (classificações NEUTRAS incorretas)
        texto_para_ia = ev.evidencia[:350] 
        vetor_ev = motor_semantico.encode(texto_para_ia)
        
        diferenca = np.abs(vetor_claim - vetor_ev)
        produto = vetor_claim * vetor_ev
        entrada_ia = np.concatenate([vetor_claim, vetor_ev, diferenca, produto]).reshape(1, -1)
        
        predicao_num = modelo_mlp.predict(entrada_ia)[0]
        ev.label = MAPA_LABELS[predicao_num]
        
        evidencias_classificadas.append(ev)
        
    return {
        "status": "sucesso",
        "claim": resultado["claim"],
        "evidencias": evidencias_classificadas
    }

# teste
if __name__ == "__main__":
    pergunta = input("Digite a alegação médica: ")
    analise = analisar_alegacao(pergunta)
    
    if analise["status"] == "sucesso":
        
        for ev in analise["evidencias"]:
            print(f"\n[CLASSIFICAÇÃO: {ev.label}] - {ev.titulo}")
            print(f"Link PubMed: {ev.link}")
            print(f"Trecho: {ev.evidencia[:200]}...") 
    else:
        print("\n[ERRO]", analise["mensagem"])