from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ncbi_api_key: str = ""
    ncbi_email: str = "contato@example.com"
    model_path: str = "/app/data/processed/modelo_mlp_profilaxia.pkl"
    sbert_model: str = "all-mpnet-base-v2"
    marian_model: str = "Helsinki-NLP/opus-mt-ROMANCE-en"
    pmids_por_busca: int = 20
    max_chars_pergunta: int = 300
    min_palavras_pergunta: int = 2
    cors_origins: str = "http://localhost:5173,http://localhost:8080"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
