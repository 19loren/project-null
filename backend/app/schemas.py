from pydantic import BaseModel


class VerificacaoRequest(BaseModel):
    pergunta: str


class EvidenciaOut(BaseModel):
    pmid: str
    pmcid: str
    titulo: str
    autores: str
    ano: str
    fonte: str = "PubMed"
    link: str
    trecho: str


class EvidenciasAgrupadas(BaseModel):
    refutam: list[EvidenciaOut]
    suportam: list[EvidenciaOut]
    neutras: list[EvidenciaOut]


class VerificacaoResponse(BaseModel):
    pergunta: str
    pergunta_traduzida: str
    claim: str
    total: int
    evidencias: EvidenciasAgrupadas


class ExemplosResponse(BaseModel):
    exemplos: list[str]


class HealthResponse(BaseModel):
    status: str


class ErroDetalhe(BaseModel):
    tipo: str
    mensagem: str


class ErroResponse(BaseModel):
    erro: ErroDetalhe
