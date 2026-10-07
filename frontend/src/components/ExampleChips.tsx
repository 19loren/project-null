import { useExemplos } from "../hooks/useVerificacao";

interface Props {
  onPick: (pergunta: string) => void;
}

export default function ExampleChips({ onPick }: Props) {
  const { data } = useExemplos();
  if (!data?.length) return null;
  return (
    <div className="examples">
      <p className="examples__title">Exemplos de perguntas</p>
      <div className="examples__list">
        {data.map((ex) => (
          <button key={ex} type="button" className="chip" onClick={() => onPick(ex)}>
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
