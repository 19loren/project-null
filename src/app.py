import os
import re
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
from buscador import coletar_evidencias, traduzir_Marian, pergunta_para_claim

print("Carregando modelo...")
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BASE = os.path.dirname(DIRETORIO_ATUAL)
caminho_modelo = os.path.join(DIRETORIO_BASE, "data", "processed", "modelo_mlp_profilaxia.pkl")

modelo_mlp = joblib.load(caminho_modelo)
motor_semantico = SentenceTransformer('all-mpnet-base-v2')
MAPA_LABELS = {0: "SUPPORTED", 1: "REFUTED", 2: "NEUTRAL"}

MAX_CHARS = 300
MIN_PALAVRAS = 2
EXEMPLO = "Tente algo como: 'A vitamina D previne a COVID-19?'"

def _erro(tipo, mensagem):
    return {"status": "erro", "tipo": tipo, "mensagem": mensagem}


def validar_pergunta(pergunta_pt):
    """Devolve (pergunta_en, None) se a entrada é válida,
    ou (None, erro) se não for."""
    texto = " ".join((pergunta_pt or "").split())   # tira espaços extras

    # 1. vazia
    if not texto:
        return None, _erro("vazia", "Digite uma pergunta para começar.")

    # 2. só números ou símbolos
    letras = re.findall(r"[^\W\d_]", texto)
    if not letras:
        return None, _erro("sem_letras",
            f"Sua entrada tem só números ou símbolos. {EXEMPLO}")

    # 3. símbolos/números demais em relação às letras
    if len(letras) / len(texto.replace(" ", "")) < 0.6:
        return None, _erro("muitos_simbolos",
            f"Sua entrada tem símbolos ou números demais para ser uma pergunta. {EXEMPLO}")

    # 4. longa demais
    if len(texto) > MAX_CHARS:
        return None, _erro("muito_longa",
            f"A pergunta passou de {MAX_CHARS} caracteres. Reescreva de forma mais curta.")

    # 5. curta demais
    palavras = re.findall(r"[^\W\d_]+", texto)
    if len(palavras) < MIN_PALAVRAS:
        return None, _erro("curta_demais",
            f"A pergunta está curta demais. {EXEMPLO}")

    # 6. sem sentido (letras repetidas, sequências de consoantes, palavra gigante)
    minusc = texto.lower()
    if (re.search(r"(.)\1{3,}", minusc)
            or re.search(r"[bcdfghjklmnpqrstvwxzç]{6,}", minusc)
            or any(len(p) > 25 for p in palavras)):
        return None, _erro("sem_sentido",
            f"Não consegui entender sua entrada. {EXEMPLO}")

    # 7. pergunta não é de sim ou não (precisa da tradução)
    pergunta_en = traduzir_Marian(texto)
    if pergunta_para_claim(pergunta_en) is None:
        return None, _erro("pergunta_invalida",
            "Não consegui transformar sua pergunta em uma afirmação a ser verificada. "
            "O sistema trabalha com perguntas de sim ou não. "
            "Em vez de 'Qual o melhor tratamento para a COVID-19?', "
            "pergunte, por exemplo, 'A ivermectina trata a COVID-19?'.")

    return pergunta_en, None

def analisar_alegacao(pergunta_pt):
    
    print(f"\n[1] Validando e traduzindo '{pergunta_pt}'...")
    pergunta_en, erro = validar_pergunta(pergunta_pt)
    if erro:
        return erro
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
# teste
if __name__ == "__main__":
    while True:
        pergunta = input("\nDigite a alegação médica (ou 'sair'): ")
        if pergunta.strip().lower() == "sair":
            break

        analise = analisar_alegacao(pergunta)

        if analise["status"] == "sucesso":

            for ev in analise["evidencias"]:
                print(f"\n[CLASSIFICAÇÃO: {ev.label}] - {ev.titulo}")
                print(f"Link PubMed: {ev.link}")
                print(f"Trecho: {ev.evidencia[:200]}...")
        else:
            print("\n[ERRO]", analise["mensagem"])