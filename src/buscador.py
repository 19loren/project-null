"""
buscador de evidencias

fluxo:
1. pega a pergunta (ja em ingles);
2. transforma em uma afirmaçao (claim);
3. extrai palavras-chave da claim;
4. busca PMIDs no PubMed a partir das palavras-chave;
5. descobre quais PMIDs tem PMCID;
6. a partir do PMCID, baixa o artigo (BioC);
7. pega a conclusao (1º paragrafo, cortado por frase ate 900 caracteres);
8. filtra por relevancia (keyword precisa aparecer na conclusao);
9. busca os dados bibliograficos (titulo, autores, ano);
10. devolve uma lista de evidencia, com `label` vazio para o modelo preencher.

uso:
    from buscador import coletar_evidencias
    resultado = coletar_evidencias("Does vitamin D prevent COVID-19?")
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import requests
import spacy
import yake
from transformers import MarianMTModel, MarianTokenizer


# tradutor
NOME_MARIAN = "Helsinki-NLP/opus-mt-ROMANCE-en"
tok_marian = MarianTokenizer.from_pretrained(NOME_MARIAN)
modelo_marian = MarianMTModel.from_pretrained(NOME_MARIAN)


def traduzir_Marian(texto: str) -> str:
    entrada = tok_marian([texto], return_tensors="pt", padding=True)
    saida = modelo_marian.generate(**entrada, max_new_tokens=256)
    return tok_marian.decode(saida[0], skip_special_tokens=True)


# pergunta -> claim
# as perguntas precisam ser de sim ou não. Se não forem, devolve None
# (a interface pode pedir outra pergunta, ou mostrar só as evidências).
nlp = spacy.load("en_core_web_sm")


def pergunta_para_claim(pergunta):
    doc = nlp(pergunta.strip().rstrip("?"))
    aux = doc[0].text.lower()
    sujeitos = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
    if not sujeitos:
        return None  # nao é pergunta de sim ou nao
    sub = list(sujeitos[0].subtree)
    fim = sub[-1].i
    sujeito = doc[sub[0].i: fim + 1].text
    resto = doc[fim + 1:].text
    if aux in {"do", "does", "did"}:
        claim = f"{sujeito} {resto}"
    else:
        claim = f"{sujeito} {aux} {resto}"
    return claim[0].upper() + claim[1:] + "."


# palavras-chave da claim: poucas palavras-chave sao melhores pq a busca é por AND
def palavrachaveEv(claim, max_k=5):
    ext = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, windowsSize=2, top=max_k)
    keywords = ext.extract_keywords(claim)
    keywords = sorted(keywords, key=lambda x: x[1])
    return [k for k, _ in keywords[:max_k]]


# busca no PubMed / PMC
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def buscar_pubmed(keywords, n=10):
    termo = " AND ".join(f'"{k}"' for k in keywords)
    termo += ' AND "free full text"[sb]'
    params = {
        "db": "pubmed",
        "term": termo,
        "retmax": n,
        "retmode": "json",
        "sort": "relevance",
    }
    r = requests.get(ESEARCH, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]  # lista de PMIDs (strings)


def buscar_com_fallback(keywords, n=10):
    for k in range(len(keywords), 0, -1):
        pmids = buscar_pubmed(keywords[:k], n)
        if pmids:
            return pmids, keywords[:k]
    return [], []


def pmid_para_pmcid(pmids):
    # TODO: trocar o e-mail abaixo por um e-mail real do grupo (o NCBI pede isso)
    params = {"ids": ",".join(pmids), "format": "json",
              "tool": "meu_projeto", "email": "seu@email.com"}
    r = requests.get(IDCONV, params=params, timeout=30)
    r.raise_for_status()
    mapa = {}
    for rec in r.json().get("records", []):
        if "pmcid" in rec:
            # a API pode devolver "pmid" como int; forçamos string
            # para bater com os PMIDs (strings) que vem do esearch
            mapa[str(rec["pmid"])] = rec["pmcid"]
    return mapa  # {"32623270": "PMC7314683", ...}


def baixar_bioc(pmcid):
    r = requests.get(BIOC.format(pmcid), timeout=30)
    if r.status_code != 200:
        return None  # fora do subconjunto Open Access
    try:
        dados = r.json()
    except requests.exceptions.JSONDecodeError:
        return None
    if isinstance(dados, list):  # as vezes vem embrulhado numa lista
        dados = dados[0]
    return dados["documents"][0]["passages"]


def pegar_conclusao(pmcid, max_chars=900):
    """Pega só o primeiro parágrafo marcado como CONCL (o resumo direto)
    e corta por frase até max_chars -- evita histórico de publicação /
    discussão longa demais (comum em revisões tipo Cochrane) e mantém o
    tamanho parecido com o das evidências do dataset de treino."""
    passagens = baixar_bioc(pmcid)
    if not passagens:
        return None

    concl = [
        p["text"].strip() for p in passagens
        if p.get("infons", {}).get("section_type") == "CONCL"
        and p.get("infons", {}).get("type") == "paragraph"
        and p.get("text", "").strip()
    ]
    if not concl:
        return None

    texto = concl[0]
    if len(texto) > max_chars:
        doc = nlp(texto)
        frases, acumulado = [], ""
        for s in doc.sents:
            if len(acumulado) + len(s.text) > max_chars:
                break
            frases.append(s.text)
            acumulado += s.text
        texto = " ".join(frases)
    return texto


def dados_bibliograficos(pmids):
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
    r = requests.get(ESUMMARY, params=params, timeout=30)
    r.raise_for_status()
    res = r.json()["result"]
    linhas = []
    for pmid in pmids:
        d = res.get(pmid)
        if not d:
            continue
        doi = next((a["value"] for a in d.get("articleids", [])
                    if a["idtype"] == "doi"), "")
        linhas.append({
            "pmid": pmid,
            "titulo": d.get("title", ""),
            "autores": "; ".join(a["name"] for a in d.get("authors", [])),
            "revista": d.get("source", ""),
            "data": d.get("pubdate", ""),
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return pd.DataFrame(linhas)


# filtro de relevancia
# a busca casa as keywords em qualquer parte do artigo, o que nao garante que
# a *conclusão* seja sobre o mesmo assunto da claim

def relevante(texto, keywords):
    # exige que pelo menos uma keyword apareça no texto da conclusao
    # é um filtro lexico, nao semantico -> limitaçao conhecida
    texto_lower = texto.lower()
    return any(k.lower() in texto_lower for k in keywords)


# lista de evidencia
@dataclass
class Evidencia:
    titulo: str
    autores: str
    ano: str
    pmid: str
    pmcid: str
    evidencia: str
    link: str = field(init=False)
    label: Optional[str] = None  # o modelo preenche depois

    def __post_init__(self):
        self.link = f"https://pmc.ncbi.nlm.nih.gov/articles/{self.pmcid}/"


def montar_lista_evidencias(brutas):
    # brutas = lista de {"pmid", "pmcid", "texto"} ja filtrada por relevancia
    # devolve uma lista de evidencia (so quem tem PMCID e conclusao)
    if not brutas:
        return []

    # uma unica chamada ao PubMed para todos os PMIDs validos
    biblio = dados_bibliograficos([e["pmid"] for e in brutas])
    por_pmid = {row["pmid"]: row for _, row in biblio.iterrows()}

    lista = []
    for e in brutas:
        b = por_pmid.get(e["pmid"])
        if b is None:
            continue
        m = re.match(r"\d{4}", b["data"])  # "2020 Jun 25" -> "2020"
        lista.append(Evidencia(
            titulo=b["titulo"],
            autores=b["autores"],
            ano=m.group(0) if m else "",
            pmid=e["pmid"],
            pmcid=e["pmcid"],
            evidencia=e["texto"],
        ))
    return lista


# func principal do buscador
def coletar_evidencias(pergunta, n=15):
    claim = pergunta_para_claim(pergunta)
    if not claim:
        return None

    kws = palavrachaveEv(claim)
    pmids, kws_usadas = buscar_com_fallback(kws, n=n)
    if not pmids:
        return None

    time.sleep(0.4)
    mapa = pmid_para_pmcid(pmids)

    brutas = []                      # conclusoes relevantes (sem dados bibliograficos ainda)
    descartadas_download = 0         # fora do Open Access / sem seção CONCL
    descartadas_irrelevantes = 0

    for pmid in pmids:               # mantem a ordem de relevancia do PubMed
        pmcid = mapa.get(pmid)
        if not pmcid:
            continue
        time.sleep(0.4)
        texto = pegar_conclusao(pmcid)
        if not texto:
            descartadas_download += 1
            continue
        if not relevante(texto, kws_usadas):
            descartadas_irrelevantes += 1
            continue
        brutas.append({"pmid": pmid, "pmcid": pmcid, "texto": texto})

    time.sleep(0.4)
    lista = montar_lista_evidencias(brutas)

    return {
        "claim": claim,
        "keywords": kws_usadas,
        "evidencias": lista,         # lista de evidencia
        "total_pmids_buscados": len(pmids),
        "descartadas_download": descartadas_download,
        "descartadas_irrelevantes": descartadas_irrelevantes,
    }


# teste pelo terminal
if __name__ == "__main__":
    pergunta = input("Pergunta: ")
    question = traduzir_Marian(pergunta)
    resultado = coletar_evidencias(question, n=15)

    if resultado is None:
        print("Não foi possível montar a claim ou não houve resultados.")
    else:
        print(resultado["claim"], resultado["keywords"])
        print(f"buscados={resultado['total_pmids_buscados']}  "
              f"sem_download={resultado['descartadas_download']}  "
              f"irrelevantes={resultado['descartadas_irrelevantes']}  "
              f"validas={len(resultado['evidencias'])}")

        for ev in resultado["evidencias"]:
            print("---", ev.pmid, ev.pmcid, ev.ano, "---")
            print(ev.titulo)
            print(ev.autores)
            print(ev.link)
            print(ev.evidencia[:300], "...")
            print("label:", ev.label, "\n")

        # para a interface (JSON):
        # print(json.dumps([asdict(e) for e in resultado["evidencias"]],
        #                  ensure_ascii=False, indent=2))

# para o modelo
# entradas
# claim -> pergunta_para_claim(question)
# evidence -> ev.evidencia
# saída
# label -> ev.label