import { useState, type FormEvent } from "react";
import { SearchIcon } from "./icons";

interface Props {
  initialValue?: string;
  onSubmit: (pergunta: string) => void;
  loading?: boolean;
  buttonLabel?: string | null;
  inputId?: string;
}

export default function SearchForm({ initialValue = "", onSubmit, loading, buttonLabel, inputId = "pergunta" }: Props) {
  const [valor, setValor] = useState(initialValue);

  const handle = (e: FormEvent) => {
    e.preventDefault();
    onSubmit(valor);
  };

  return (
    <form className="search-form" onSubmit={handle}>
      <div className="input-wrap">
        <input
          id={inputId}
          className="input"
          type="text"
          value={valor}
          onChange={(e) => setValor(e.target.value)}
          placeholder="Ex.: A vitamina D previne a COVID-19?"
          autoComplete="off"
        />
        <button type="submit" className="input-wrap__icon" aria-label="Pesquisar" disabled={loading}>
          <SearchIcon />
        </button>
      </div>
      {buttonLabel && (
        <button type="submit" className="btn-primary" disabled={loading}>
          {buttonLabel}
        </button>
      )}
    </form>
  );
}

export { type Props as SearchFormProps };
