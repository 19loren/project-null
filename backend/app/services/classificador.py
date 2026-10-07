"""SBERT + MLP: classifica cada evidência frente à claim."""
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.services.pubmed import Evidencia

MAPA_LABELS = {0: "SUPPORTED", 1: "REFUTED", 2: "NEUTRAL"}
MAX_CHARS_EVIDENCIA = 350  # mesmo corte usado na versão original


class Classificador:
    def __init__(self) -> None:
        self._mlp = joblib.load(settings.model_path)
        self._sbert = SentenceTransformer(settings.sbert_model)

    def classificar(self, claim: str, evidencias: list[Evidencia]) -> list[Evidencia]:
        if not evidencias:
            return evidencias
        v_claim = self._sbert.encode(claim)
        v_ev = self._sbert.encode([e.texto[:MAX_CHARS_EVIDENCIA] for e in evidencias])
        v_claim = np.broadcast_to(v_claim, v_ev.shape)
        entrada = np.hstack([v_claim, v_ev, np.abs(v_claim - v_ev), v_claim * v_ev])
        for ev, pred in zip(evidencias, self._mlp.predict(entrada)):
            ev.label = MAPA_LABELS[int(pred)]
        return evidencias
