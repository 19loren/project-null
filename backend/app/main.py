import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.errors import register_error_handlers
from app.routers import verificacoes
from app.services.verificador import Verificador

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Carregando modelos...")
    app.state.verificador = Verificador()
    logging.info("Modelos prontos.")
    yield


app = FastAPI(title="ProfilaxIA API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
register_error_handlers(app)
app.include_router(verificacoes.router)
