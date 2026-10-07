import { useNavigate, useSearchParams } from "react-router-dom";
import EvidenceGroup from "../components/EvidenceGroup";
import Header from "../components/Header";
import SearchForm from "../components/SearchForm";
import { useVerificacao } from "../hooks/useVerificacao";
import { ApiRequestError } from "../api/client";

export default function ResultadosPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const q = (params.get("q") ?? "").trim();
  const { data, error, isFetching } = useVerificacao(q);

  const buscar = (pergunta: string) => navigate(`/resultados?q=${encodeURIComponent(pergunta.trim())}`);
  const mensagem =
    error instanceof ApiRequestError ? error.erro.mensagem : error ? "Ocorreu um erro inesperado." : null;
  // sem q, a validação do backend devolve a mensagem "Digite uma pergunta..."
  const semPergunta = q.length === 0;

  return (
    <>
      <div className="results-shell">
        <Header variant="results" />
        <div className="results-search">
          <label htmlFor="pergunta-resultados" className="results-search__label">Digite sua dúvida</label>
          <SearchForm key={q} initialValue={q} onSubmit={buscar} loading={isFetching} inputId="pergunta-resultados" />
        </div>

        {semPergunta && <p className="notice">Digite uma pergunta para começar.</p>}

        {isFetching && (
          <div className="notice" role="status">
            <span className="spinner" /> Buscando e classificando evidências isso pode levar alguns segundos.
          </div>
        )}

        {!isFetching && mensagem && <div className="notice notice--error" role="alert">{mensagem}</div>}

        {!isFetching && data && (
          <div className="groups">
            <EvidenceGroup tom="refuta" titulo="Evidências que refutam" evidencias={data.evidencias.refutam} />
            <EvidenceGroup tom="suporta" titulo="Evidências que suportam" evidencias={data.evidencias.suportam} />
            <EvidenceGroup tom="neutra" titulo="Evidências neutras" evidencias={data.evidencias.neutras} />
          </div>
        )}
      </div>
    </>
  );
}
