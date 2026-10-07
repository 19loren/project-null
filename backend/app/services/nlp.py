"""Tradução PT→EN, pergunta→claim e palavras-chave."""
import re

import spacy
import yake
from transformers import MarianMTModel, MarianTokenizer

from app.config import settings


class Tradutor:
    def __init__(self) -> None:
        self._tok = MarianTokenizer.from_pretrained(settings.marian_model)
        self._modelo = MarianMTModel.from_pretrained(settings.marian_model)

    def traduzir(self, texto: str) -> str:
        entrada = self._tok([texto], return_tensors="pt", padding=True)
        saida = self._modelo.generate(**entrada, max_new_tokens=256)
        return self._tok.decode(saida[0], skip_special_tokens=True)


nlp = spacy.load("en_core_web_sm")


def _terceira_pessoa(verbo: str) -> str:
    v = verbo.lower()
    if v.endswith(("s", "x", "z", "ch", "sh", "o")):
        return v + "es"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def _finalizar(claim: str) -> str:
    claim = " ".join(claim.split())
    return claim[0].upper() + claim[1:] + "."


def pergunta_para_claim(pergunta: str) -> str | None:
    """Converte pergunta de sim/não em afirmação; None se não for esse tipo.

    Aceita (1) perguntas com auxiliar inicial ("Does X reduce Y?") e
    (2) afirmações em forma de pergunta ("X reduces Y?"), comuns na tradução PT→EN.
    """
    doc = nlp(pergunta.strip().rstrip("?"))
    if len(doc) == 0 or doc[0].tag_ in ("WP", "WRB", "WDT", "WP$"):
        return None  # perguntas abertas (what/which/how...) não são de sim ou não

    if doc[0].pos_ != "AUX":
        # "Ivermectin treats COVID-19?" -> já é uma afirmação, se tiver sujeito e verbo
        tem_sujeito = any(t.dep_ in ("nsubj", "nsubjpass", "csubj") for t in doc)
        tem_verbo = any(t.pos_ in ("VERB", "AUX") and t.dep_ == "ROOT" for t in doc)
        if tem_sujeito and tem_verbo:
            return _finalizar(doc.text)
        return None

    aux = doc[0].text.lower()
    sujeitos = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass")]
    if not sujeitos:
        return None
    sub = list(sujeitos[0].subtree)
    fim = sub[-1].i
    sujeito = doc[max(sub[0].i, 1) : fim + 1].text
    resto_tokens = list(doc[fim + 1 :])
    if aux in {"do", "does", "did"}:
        # o verbo vem no infinitivo após o auxiliar: "does X reduce" -> "X reduces"
        if aux != "did" and resto_tokens and resto_tokens[0].tag_ == "VB":
            resto_tokens[0:1] = []
            resto = " ".join([_terceira_pessoa(doc[fim + 1].text)] + [t.text for t in resto_tokens])
        else:
            resto = doc[fim + 1 :].text
        claim = f"{sujeito} {resto}"
    else:
        claim = f"{sujeito} {aux} {doc[fim + 1 :].text}"
    return _finalizar(claim)


_TERMO_COVID = re.compile(r"\b(covid(?:-?19)?|sars-cov-2|coronavirus)\b", re.I)


def palavras_chave(claim: str, max_k: int = 5) -> list[str]:
    ext = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, windowsSize=2, top=max_k)
    kws = [k for k, _ in sorted(ext.extract_keywords(claim), key=lambda x: x[1])][:max_k]
    # o YAKE às vezes descarta o tema ("COVID-19"); sem ele a busca perde o contexto.
    # Entra na 2ª posição: a busca com fallback remove termos do fim para o começo.
    m = _TERMO_COVID.search(claim)
    if m and not any(_TERMO_COVID.fullmatch(k) for k in kws):
        kws = kws[:max_k - 1]
        kws.insert(min(1, len(kws)), m.group(0))
    return kws
