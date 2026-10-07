"""Coleta de evidências no PubMed / PMC (porte do src/buscador.py)."""
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

from app.config import settings
from app.errors import ApiError
from app.services.nlp import nlp, palavras_chave

log = logging.getLogger(__name__)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI: 10 req/s com chave, 3 req/s sem. Margem de segurança.
_INTERVALO = 0.12 if settings.ncbi_api_key else 0.4
_lock = threading.Lock()
_ultimo = 0.0


def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET com rate limit global e chave NCBI injetada (só nos endpoints E-utilities)."""
    global _ultimo
    params = dict(params or {})
    params.update(tool="profilaxia", email=settings.ncbi_email)
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    with _lock:
        espera = _INTERVALO - (time.monotonic() - _ultimo)
        if espera > 0:
            time.sleep(espera)
        _ultimo = time.monotonic()
    return requests.get(url, params=params, timeout=30)


@dataclass
class Evidencia:
    titulo: str
    autores: str
    ano: str
    pmid: str
    pmcid: str
    texto: str
    label: str | None = None

    @property
    def link(self) -> str:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{self.pmcid}/"


def _buscar_pubmed(keywords: list[str], n: int) -> list[str]:
    termo = " AND ".join(f'"{k}"' for k in keywords) + ' AND "free full text"[sb]'
    r = _get(ESEARCH, {"db": "pubmed", "term": termo, "retmax": n, "retmode": "json", "sort": "relevance"})
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def _buscar_com_fallback(keywords: list[str], n: int) -> tuple[list[str], list[str]]:
    for k in range(len(keywords), 0, -1):
        pmids = _buscar_pubmed(keywords[:k], n)
        if pmids:
            return pmids, keywords[:k]
    return [], []


class PmcIndisponivel(Exception):
    """O PMC respondeu 429/5xx mesmo após novas tentativas."""


def _baixar_artigo_xml(pmcid: str) -> ET.Element | None:
    """Baixa o texto completo (JATS) via efetch. None se indisponível para esse artigo."""
    for tentativa in range(2):
        r = _get(EFETCH, {"db": "pmc", "id": pmcid, "retmode": "xml"})
        if r.status_code == 200:
            break
        if r.status_code not in (429, 500, 502, 503, 504):
            return None
        time.sleep(1.0 + tentativa)
    else:
        raise PmcIndisponivel(pmcid)
    try:
        return ET.fromstring(r.content).find("article")
    except ET.ParseError:
        return None


def _primeiro_paragrafo(sec: ET.Element) -> str | None:
    for p in sec.findall("p"):
        texto = " ".join("".join(p.itertext()).split())
        if texto:
            return texto
    return None


def _extrair_conclusao(artigo: ET.Element) -> str | None:
    """1º parágrafo da seção de conclusão (corpo; senão, resumo estruturado)."""
    def eh_conclusao(sec: ET.Element) -> bool:
        titulo = (sec.findtext("title") or "").lower()
        return "conclu" in titulo or "conclu" in (sec.get("sec-type") or "").lower()

    escopos = [artigo.find("body"), artigo.find("front/article-meta/abstract")]
    for escopo in escopos:
        if escopo is None:
            continue
        for sec in escopo.iter("sec"):
            if eh_conclusao(sec):
                texto = _primeiro_paragrafo(sec)
                if texto:
                    return texto
    return None


def _pegar_conclusao(pmcid: str, max_chars: int = 900) -> str | None:
    artigo = _baixar_artigo_xml(pmcid)
    if artigo is None:
        return None
    texto = _extrair_conclusao(artigo)
    if not texto:
        return None
    if len(texto) > max_chars:
        frases, acumulado = [], ""
        for s in nlp(texto).sents:
            if len(acumulado) + len(s.text) > max_chars:
                break
            frases.append(s.text)
            acumulado += s.text
        texto = " ".join(frases)
    return texto or None


def _relevante(texto: str, keywords: list[str]) -> bool:
    t = texto.lower()
    return any(k.lower() in t for k in keywords)


def _dados_bibliograficos(pmids: list[str]) -> dict[str, dict]:
    r = _get(ESUMMARY, {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
    r.raise_for_status()
    res = r.json()["result"]
    out = {}
    for pmid in pmids:
        d = res.get(pmid)
        if d:
            out[pmid] = {
                "pmcid": next((a["value"] for a in d.get("articleids", []) if a["idtype"] == "pmc"), ""),
                "titulo": d.get("title", ""),
                "autores": "; ".join(a["name"] for a in d.get("authors", [])),
                "data": d.get("pubdate", ""),
            }
    return out


def coletar_evidencias(claim: str, n: int | None = None) -> list[Evidencia]:
    n = n or settings.pmids_por_busca
    try:
        kws = palavras_chave(claim)
        pmids, kws_usadas = _buscar_com_fallback(kws, n)
        if not pmids:
            return []
        biblio = _dados_bibliograficos(pmids)  # já traz o PMCID de cada artigo

        lista = []
        falhas_seguidas = 0
        for pmid in pmids:  # mantém a ordem de relevância do PubMed
            b = biblio.get(pmid)
            if not b or not b["pmcid"]:
                continue
            try:
                texto = _pegar_conclusao(b["pmcid"])
                falhas_seguidas = 0
            except PmcIndisponivel:
                falhas_seguidas += 1
                if falhas_seguidas >= 2:  # serviço fora do ar: não vale esperar os demais
                    raise ApiError(
                        502, "pubmed_indisponivel",
                        "O serviço de artigos do PubMed Central está indisponível agora. Tente novamente em alguns minutos.",
                    )
                continue
            if not texto or not _relevante(texto, kws_usadas):
                continue
            m = re.match(r"\d{4}", b["data"])
            lista.append(Evidencia(b["titulo"], b["autores"], m.group(0) if m else "", pmid, b["pmcid"], texto))
        return lista
    except requests.RequestException as exc:
        log.exception("Falha ao consultar NCBI")
        raise ApiError(502, "pubmed_indisponivel", "Não foi possível consultar o PubMed agora. Tente novamente.") from exc
