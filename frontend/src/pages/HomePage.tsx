import { useState } from "react";
import { useNavigate } from "react-router-dom";
import ExampleChips from "../components/ExampleChips";
import Header from "../components/Header";
import { ShieldIcon } from "../components/icons";
import SearchForm from "../components/SearchForm";

export default function HomePage() {
  const navigate = useNavigate();
  const [valor, setValor] = useState("");
  // `key` força o SearchForm a remontar com o texto do exemplo escolhido
  const [formKey, setFormKey] = useState(0);

  const buscar = (pergunta: string) => {
    const q = pergunta.trim();
    navigate(`/resultados?q=${encodeURIComponent(q)}`);
  };

  return (
    <>
      <Header variant="home" />
      <main className="home">
        <span className="pill">
          <ShieldIcon width={14} height={14} />
          Pesquisa científica em saúde. Não substitui orientação médica.
        </span>
        <h1 className="home__title">Verifique uma dúvida com evidências científicas sobre covid-19</h1>
        <p className="home__subtitle">Digite uma pergunta sobre saúde e encontre evidências científicas relacionadas.</p>

        <div className="card home__card">
          <label htmlFor="pergunta" className="label">Sua pergunta</label>
          <SearchForm
            key={formKey}
            initialValue={valor}
            onSubmit={buscar}
            buttonLabel="Verificar evidências →"
          />
          <ExampleChips
            onPick={(ex) => {
              setValor(ex);
              setFormKey((k) => k + 1);
            }}
          />
        </div>
        <p className="home__footnote">A pergunta será utilizada como contexto para iniciar a busca por evidências científicas.</p>
      </main>
    </>
  );
}
