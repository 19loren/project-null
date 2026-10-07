from fastapi import APIRouter, Request

from app.schemas import (
    ErroResponse,
    ExemplosResponse,
    HealthResponse,
    VerificacaoRequest,
    VerificacaoResponse,
)

router = APIRouter(prefix="/api/v1")

EXEMPLOS = [
    "Can vaccinated people still get COVID-19?",
    "Can wearing a mask reduce the spread of COVID-19",
    "Can antibiotics treat COVID-19?",
]


@router.get("/health", response_model=HealthResponse, tags=["infra"])
def health():
    return HealthResponse(status="ok")


@router.get("/exemplos", response_model=ExemplosResponse, tags=["verificações"])
def exemplos():
    return ExemplosResponse(exemplos=EXEMPLOS)


@router.post(
    "/verificacoes",
    response_model=VerificacaoResponse,
    responses={
        404: {"model": ErroResponse, "description": "Nenhuma evidência encontrada"},
        422: {"model": ErroResponse, "description": "Pergunta inválida"},
        502: {"model": ErroResponse, "description": "PubMed indisponível"},
    },
    tags=["verificações"],
    summary="Verifica uma pergunta de sim/não contra evidências científicas",
)
def criar_verificacao(body: VerificacaoRequest, request: Request):
    # `def` (não async): roda no threadpool, pois o pipeline é bloqueante (I/O + CPU).
    return request.app.state.verificador.verificar(body.pergunta)
