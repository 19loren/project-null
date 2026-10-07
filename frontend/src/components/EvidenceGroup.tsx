import type { Evidencia } from "../types/api";
import { CheckIcon, ChevronIcon, DocIcon, MinusIcon, XIcon } from "./icons";

export type Tom = "refuta" | "suporta" | "neutra";

const ICONE = { refuta: XIcon, suporta: CheckIcon, neutra: MinusIcon } as const;

interface Props {
  tom: Tom;
  titulo: string;
  evidencias: Evidencia[];
}

export default function EvidenceGroup({ tom, titulo, evidencias }: Props) {
  const Icone = ICONE[tom];
  return (
    <section className="card group">
      <h2 className="group__title">
        <span className={`badge badge--${tom}`}><Icone width={22} height={22} /></span>
        {titulo}
      </h2>
      {evidencias.length === 0 ? (
        <p className="group__empty">Nenhuma evidência encontrada nesta categoria.</p>
      ) : (
        <ul className="evidence-list">
          {evidencias.map((ev) => (
            <li key={ev.pmcid}>
              <a className="evidence" href={ev.link} target="_blank" rel="noopener noreferrer" title={ev.trecho}>
                <DocIcon className="evidence__icon" width={26} height={26} />
                <span className="evidence__body">
                  <span className="evidence__title">{ev.titulo}</span>
                  <span className="evidence__meta">{[ev.fonte, ev.ano].filter(Boolean).join(" · ")}</span>
                </span>
                <ChevronIcon className="evidence__chevron" width={18} height={18} />
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
