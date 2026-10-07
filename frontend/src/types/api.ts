export interface Evidencia {
  pmid: string;
  pmcid: string;
  titulo: string;
  autores: string;
  ano: string;
  fonte: string;
  link: string;
  trecho: string;
}

export interface EvidenciasAgrupadas {
  refutam: Evidencia[];
  suportam: Evidencia[];
  neutras: Evidencia[];
}

export interface VerificacaoResponse {
  pergunta: string;
  pergunta_traduzida: string;
  claim: string;
  total: number;
  evidencias: EvidenciasAgrupadas;
}

export interface ExemplosResponse {
  exemplos: string[];
}

export interface ErroApi {
  tipo: string;
  mensagem: string;
}
