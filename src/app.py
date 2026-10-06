import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from buscador import coletar_evidencias, traduzir_Marian

print("carregando motores...")
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_BASE = os.path.dirname(DIRETORIO_ATUAL)
caminho_modelo = os.path.join(DIRETORIO_BASE, "data", "processed", "modelo_mlp_profilaxia.pkl")

# carrega a rede neural e motor semantico
modelo_mlp = joblib.load(caminho_modelo)
motor_semantico = SentenceTransformer('all-mpnet-base-v2')

MAPA_LABELS = {
    0: "SUPPORTED",
    1: "NEUTRAL",
    2: "REFUTED"
}

def classificar_pergunta(pergunta_pt):
    print(f"\n[1] A traduzir e buscar no PubMed: '{pergunta_pt}'...")
    
    # traduz (opcional se quiser usar o da colega direto) e busca
    pergunta_en = traduzir_Marian(pergunta_pt)
    resultado = coletar_evidencias(pergunta_en, n=10)
    
    if not resultado or not resultado["evidencias"]:
        return {"status": "erro", "mensagem": "Não foram encontradas evidências suficientes."}
    
    # gera o vetor da claim
    vetor_claim = motor_semantico.encode(resultado["claim"])
    
    evidencias_finais = []
    
    print(f"[2] A analisar {len(resultado['evidencias'])} artigos científicos...")
    
    # itera sobre a lista de objetos "evidencia"
    for ev in resultado["evidencias"]:
        # transforma o texto do artigo num vetor matematico
        vetor_ev = motor_semantico.encode(ev.evidencia)
        
        # a matematica usada no prep.py
        diferenca = np.abs(vetor_claim - vetor_ev)
        produto = vetor_claim * vetor_ev
        
        # empacota (3072 dimensoes)
        entrada = np.concatenate([vetor_claim, vetor_ev, diferenca, produto]).reshape(1, -1)
        
        # faz a previsao (0, 1 ou 2)
        predicao_num = modelo_mlp.predict(entrada)[0]
        
        # preenche o espaço vazio (None)
        ev.label = MAPA_LABELS[predicao_num]
        
        evidencias_finais.append(ev)
    
    return {
        "status": "sucesso",
        "claim_en": resultado["claim"],
        "evidencias": evidencias_finais
    }

# teste
if __name__ == "__main__":
    pergunta_usuario = "A vitamina D previne a COVID-19?"
    analise = classificar_pergunta(pergunta_usuario)
    
    if analise["status"] == "sucesso":
        print("\n" + "="*50)
        print(f"CLAIM: {analise['claim_en']}")
        print("="*50)
        
        for ev in analise["evidencias"]:
            print(f"\n[{ev.label}] - {ev.titulo}")
            print(f"Ano: {ev.ano} | PMCID: {ev.pmcid}")
            print(f"Link: {ev.link}")
            print(f"Trecho: {ev.evidencia[:200]}...")
    else:
        print(analise["mensagem"])