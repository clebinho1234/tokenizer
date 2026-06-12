import os

def carregar_corpus(pasta):
    textos = []
    
    for ficheiro in os.listdir(pasta):
        if ficheiro.endswith(".txt"):
            caminho = os.path.join(pasta, ficheiro)
            
            with open(caminho, "r", encoding="utf-8") as f:
                texto = f.read().replace("\n", " ")
                textos.append(texto)
                
    return textos