import os
import spacy
import yake
import requests
import time
import numpy as np
import joblib
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer

# carregamento dos dados
print("A iniciar os motores de IA...")

# modelos de traduçao (PT -> EN)
nome_marian = "Helsinki-NLP/opus-mt-ROMANCE-en"
tok_marian = MarianTokenizer.from_pretrained(nome_marian)
modelo_marian = MarianMTModel.from_pretrained(nome_marian)

# motor semantico e modelo supervisionado
sbert_model = SentenceTransformer('all-mpnet-base-v2')
nlp = spacy.load("en_core_web_sm")

# caminho para o modelo treinado salvo
CAMINHO_MODELO = os.path.join("data", "processed", "modelo_mlp_profilaxia.pkl")
mlp_model = joblib.load(CAMINHO_MODELO)

# funçoes do buscador
def traduzir_Marian(texto: str) -> str:
    entrada = tok_marian([texto], return_tensors="pt", padding=True)
    saida = modelo_marian.generate(**entrada, max_new_tokens=256)
    return tok_marian.decode(saida[0], skip_special_tokens=True)

def pergunta_para_claim(pergunta):
    doc = nlp(pergunta.strip().rstrip("?"))
    aux = doc[0].text.lower()
    sujeitos = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
    if not sujeitos: return pergunta  # fallback seguro
    sub = list(sujeitos[0].subtree)
    fim = sub[-1].i
    sujeito = doc[sub[0].i : fim + 1].text
    resto = doc[fim + 1 :].text
    if aux in {"do", "does", "did"}: claim = f"{sujeito} {resto}"
    else: claim = f"{sujeito} {aux} {resto}"
    return claim[0].upper() + claim[1:] + "."

def palavrachaveEv(claim, max_k=3):
    ext = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, windowsSize=2, top=max_k)
    keywords = sorted(ext.extract_keywords(claim), key=lambda x: x[1])
    return [k for k, _ in keywords[:max_k]]

def buscar_pubmed(keywords, n=10):
    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    termo = " AND ".join(f'"{k}"' for k in keywords) + ' AND "free full text"[sb]'
    r = requests.get(ESEARCH, params={"db": "pubmed", "term": termo, "retmax": n, "retmode": "json", "sort": "relevance"}, timeout=30)
    return r.json()["esearchresult"]["idlist"]

def pmid_para_pmcid(pmids):
    IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    r = requests.get(IDCONV, params={"ids": ",".join(pmids), "format": "json", "tool": "profilaxia"}, timeout=30)
    return {str(rec["pmid"]): rec["pmcid"] for rec in r.json().get("records", []) if "pmcid" in rec}

def pegar_conclusao(pmcid, max_chars=900):
    BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"
    try:
        r = requests.get(BIOC.format(pmcid), timeout=30)
        if r.status_code != 200: return None
        dados = r.json()
        if isinstance(dados, list): dados = dados[0]
        passagens = dados["documents"][0]["passages"]
        concl = [p["text"].strip() for p in passagens if p.get("infons", {}).get("section_type") == "CONCL"]
        return concl[0][:max_chars] if concl else None
    except:
        return None

def coletar_evidencias(pergunta_en, n=10):
    claim = pergunta_para_claim(pergunta_en)
    kws = palavrachaveEv(claim)
    pmids = buscar_pubmed(kws, n)
    
    if not pmids: return []
    mapa = pmid_para_pmcid(pmids)
    
    evidencias = []
    for pmid in pmids:
        pmcid = mapa.get(pmid)
        if not pmcid: continue
        time.sleep(0.4) # Respeitar limites da API
        texto = pegar_conclusao(pmcid)
        # Filtro de relevância léxica (feita pela sua colega)
        if texto and any(k.lower() in texto.lower() for k in kws):
            evidencias.append({"pmid": pmid, "pmcid": pmcid, "texto": texto})
            
    return evidencias, claim

# inferencia
def analisar_alegacao(claim_pt):
    # traduzir para ingles
    print(f"A traduzir a alegação: '{claim_pt}'...")
    claim_en = traduzir_Marian(claim_pt)
    
    # buscar evidencias
    print("A buscar evidências no PubMed...")
    resultados_busca = coletar_evidencias(claim_en, n=10)
    
    # se nao encontrar nada, cai na regra de insuficiencia
    if not resultados_busca or len(resultados_busca[0]) == 0:
        return {
            "Label": "Neutral",
            "Mensagem": "Não foram encontradas evidências suficientes para essa alegação.",
            "Links_Suporte": [],
            "Links_Refutacao": []
        }
        
    evidencias, claim_tratada = resultados_busca
    emb_c = sbert_model.encode(claim_tratada) # vetoriza a alegaçao
    
    suportes, refutacoes, neutras = [], [], []
    
    # classificar cada evidencia contra a alegaçao
    print(f"{len(evidencias)} evidências encontradas. A passar pelo classificador neural...")
    for ev in evidencias:
        emb_e = sbert_model.encode(ev["texto"]) # vetoriza a evidencia
        
        # matematica dos embeddings
        diff = np.abs(emb_c - emb_e)
        prod = emb_c * emb_e
        features = np.hstack([emb_c, emb_e, diff, prod]).reshape(1, -1)
        
        # o predict devolve 0(Suporta), 1(Refuta), 2(Neutra)
        predicao = mlp_model.predict(features)[0] 
        link_pubmed = f"https://pubmed.ncbi.nlm.nih.gov/{ev['pmid']}/"
        
        if predicao == 0:
            suportes.append(link_pubmed)
        elif predicao == 2:
            neutras.append(link_pubmed)
        else:
            refutacoes.append(link_pubmed)

    # logica final
    # contradiçao (evidencias que suportam ou refutam)
    if len(suportes) > 0 and len(refutacoes) > 0:
        return {
            "Label": "Neutral",
            "Mensagem": "Foram encontradas evidências que se contradizem.",
            "Links_Suporte": suportes,
            "Links_Refutacao": refutacoes
        }
    
    # maioria suporta
    elif len(suportes) > 0 and len(refutacoes) == 0:
        return {
            "Label": "Supported",
            "Mensagem": "A alegação é suportada pelas evidências científicas.",
            "Links_Suporte": suportes,
            "Links_Refutacao": []
        }
        
    # maioria refuta
    elif len(refutacoes) > 0 and len(suportes) == 0:
        return {
            "Label": "Refuted",
            "Mensagem": "A alegação foi refutada pelas evidências científicas.",
            "Links_Suporte": [],
            "Links_Refutacao": refutacoes
        }
        
    # apenas evidencias neutras/inconclusivas
    else:
        return {
            "Label": "Neutral",
            "Mensagem": "Não foram encontradas evidências conclusivas suficientes para classificar essa alegação de forma rigorosa.",
            "Links_Suporte": [],
            "Links_Refutacao": []
        }

# teste
if __name__ == "__main__":
    alegacao_usuario = "Tomar vitamina C em altas doses cura o vírus da covid."
    resultado_final = analisar_alegacao(alegacao_usuario)
    
    print("VEREDITO FINAL DO PROFILAXIA")
    print("="*40)
    print(f"Rótulo: {resultado_final['Label']}")
    print(f"Motivo: {resultado_final['Mensagem']}")
    print(f"Fontes que Suportam: {resultado_final['Links_Suporte']}")
    print(f"Fontes que Refutam: {resultado_final['Links_Refutacao']}")