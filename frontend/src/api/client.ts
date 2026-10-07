import axios from "axios";
import type { ErroApi } from "../types/api";

export const http = axios.create({
  baseURL: "/api/v1",
  timeout: 180_000, // a verificação consulta o PubMed e pode levar dezenas de segundos
});

export class ApiRequestError extends Error {
  constructor(public readonly erro: ErroApi) {
    super(erro.mensagem);
  }
}

http.interceptors.response.use(
  (r) => r,
  (err) => {
    const erro: ErroApi | undefined = err?.response?.data?.erro;
    if (erro) throw new ApiRequestError(erro);
    throw new ApiRequestError({
      tipo: "rede",
      mensagem: "Não foi possível conectar ao servidor. Tente novamente.",
    });
  },
);
