"""Orquestra: validação → tradução → claim → PubMed → classificação."""
from app.errors import ApiError
from app.schemas import EvidenciaOut, EvidenciasAgrupadas, VerificacaoResponse
from app.services import pubmed
from app.services.classificador import Classificador
from app.services.nlp import Tradutor, pergunta_para_claim
from app.services.validacao import normalizar_e_validar


class Verificador:
    def __init__(self) -> None:
        self.tradutor = Tradutor()
        self.classificador = Classificador()

    def verificar(self, pergunta: str) -> VerificacaoResponse:
        texto = normalizar_e_validar(pergunta)

        pergunta_en = self.tradutor.traduzir(texto)
        claim = pergunta_para_claim(pergunta_en)
        if claim is None:
            raise ApiError(
                422,
                "pergunta_invalida",
                "Não consegui transformar sua pergunta em uma afirmação a ser verificada. "
                "O sistema trabalha com perguntas de sim ou não. "
                "Em vez de 'Qual o melhor tratamento para a COVID-19?', "
                "pergunte, por exemplo, 'A ivermectina trata a COVID-19?'.",
            )

        evidencias = pubmed.coletar_evidencias(claim)
        if not evidencias:
            raise ApiError(
                404,
                "sem_evidencias",
                "Não foram encontradas evidências suficientes para essa pergunta. "
                "Tente reformulá-la com outros termos.",
            )

        self.classificador.classificar(claim, evidencias)

        grupos = {"REFUTED": [], "SUPPORTED": [], "NEUTRAL": []}
        for ev in evidencias:
            grupos[ev.label].append(
                EvidenciaOut(
                    pmid=ev.pmid,
                    pmcid=ev.pmcid,
                    titulo=ev.titulo,
                    autores=ev.autores,
                    ano=ev.ano,
                    link=ev.link,
                    trecho=ev.texto,
                )
            )
        return VerificacaoResponse(
            pergunta=texto,
            pergunta_traduzida=pergunta_en,
            claim=claim,
            total=len(evidencias),
            evidencias=EvidenciasAgrupadas(
                refutam=grupos["REFUTED"], suportam=grupos["SUPPORTED"], neutras=grupos["NEUTRAL"]
            ),
        )
