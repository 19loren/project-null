"""Validação da entrada do usuário (mesmas regras do src/app.py)."""
import re

from app.config import settings
from app.errors import ApiError

EXEMPLO = "Tente algo como: 'A vitamina D previne a COVID-19?'"


def _invalida(tipo: str, mensagem: str) -> ApiError:
    return ApiError(422, tipo, mensagem)


def normalizar_e_validar(pergunta: str) -> str:
    """Devolve o texto normalizado ou levanta ApiError(422)."""
    texto = " ".join((pergunta or "").split())

    if not texto:
        raise _invalida("vazia", "Digite uma pergunta para começar.")

    letras = re.findall(r"[^\W\d_]", texto)
    if not letras:
        raise _invalida("sem_letras", f"Sua entrada tem só números ou símbolos. {EXEMPLO}")

    if len(letras) / len(texto.replace(" ", "")) < 0.6:
        raise _invalida(
            "muitos_simbolos",
            f"Sua entrada tem símbolos ou números demais para ser uma pergunta. {EXEMPLO}",
        )

    if len(texto) > settings.max_chars_pergunta:
        raise _invalida(
            "muito_longa",
            f"A pergunta passou de {settings.max_chars_pergunta} caracteres. Reescreva de forma mais curta.",
        )

    palavras = re.findall(r"[^\W\d_]+", texto)
    if len(palavras) < settings.min_palavras_pergunta:
        raise _invalida("curta_demais", f"A pergunta está curta demais. {EXEMPLO}")

    minusc = texto.lower()
    if (
        re.search(r"(.)\1{3,}", minusc)
        or re.search(r"[bcdfghjklmnpqrstvwxzç]{6,}", minusc)
        or any(len(p) > 25 for p in palavras)
    ):
        raise _invalida("sem_sentido", f"Não consegui entender sua entrada. {EXEMPLO}")

    return texto
