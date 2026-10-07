import { useQuery } from "@tanstack/react-query";
import { listarExemplos, verificarPergunta } from "../api/verificacoes";

export function useVerificacao(pergunta: string) {
  return useQuery({
    queryKey: ["verificacao", pergunta],
    queryFn: () => verificarPergunta(pergunta),
    enabled: pergunta.length > 0,
    retry: false,
    staleTime: Infinity,
  });
}

export function useExemplos() {
  return useQuery({ queryKey: ["exemplos"], queryFn: listarExemplos, staleTime: Infinity });
}
