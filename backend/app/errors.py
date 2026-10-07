from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Erro de domínio com status HTTP e tipo estável para o frontend."""

    def __init__(self, status_code: int, tipo: str, mensagem: str):
        self.status_code = status_code
        self.tipo = tipo
        self.mensagem = mensagem


def _body(tipo: str, mensagem: str) -> dict:
    return {"erro": {"tipo": tipo, "mensagem": mensagem}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code, content=_body(exc.tipo, exc.mensagem))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, __: RequestValidationError):
        return JSONResponse(status_code=422, content=_body("payload_invalido", "Requisição inválida."))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, __: Exception):
        return JSONResponse(status_code=500, content=_body("erro_interno", "Erro interno. Tente novamente."))
