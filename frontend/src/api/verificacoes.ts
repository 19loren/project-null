import { http } from "./client";
import type { ExemplosResponse, VerificacaoResponse } from "../types/api";

export async function verificarPergunta(pergunta: string): Promise<VerificacaoResponse> {
  const { data } = await http.post<VerificacaoResponse>("/verificacoes", { pergunta });
  return data;
}

export async function listarExemplos(): Promise<string[]> {
  const { data } = await http.get<ExemplosResponse>("/exemplos");
  return data.exemplos;
}
