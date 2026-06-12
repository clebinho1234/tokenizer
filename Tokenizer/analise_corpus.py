from carregar_corpus import carregar_corpus
import nltk
import string
import matplotlib.pyplot as plt

# Necessário descarregar os recursos do NLTK (apenas na primeira execução)
# nltk.download('punkt')
# nltk.download('punkt_tab')

treino = carregar_corpus("corpus/treino")
teste = carregar_corpus("corpus/teste")

tokens_treino = nltk.word_tokenize(" ".join(treino))
tokens_teste = nltk.word_tokenize(" ".join(teste))

tokens_treino = [t for t in tokens_treino if t not in string.punctuation]
tokens_teste = [t for t in tokens_teste if t not in string.punctuation]

frequencia = nltk.FreqDist(tokens_treino + tokens_teste)

frequencia_treino = nltk.FreqDist(tokens_treino)

num_docs = len(treino) + len(teste)
num_palavras = len(tokens_treino) + len(tokens_teste)

vocabulario = set(tokens_treino)
riqueza_lexical = len(vocabulario) / len(tokens_treino)

# out of vocabulary (OOV)
oov = [t for t in tokens_teste if t not in vocabulario]

print("Número de documentos:", num_docs)
print("Número total de palavras:", num_palavras)
print("Vocabulário:", len(vocabulario))
print("Riqueza lexical:", riqueza_lexical)
print("Número de palavras OOV:", len(oov))

print("\nTop 20 palavras:")
print(frequencia.most_common(20))

print("\nExemplos OOV:")
print(list(set(oov))[:10])

raras = [palavra for palavra, freq in frequencia.items() if freq == 1]

print("\nPalavras raras (exemplos):")
print(raras[:20])

longas = [t for t in tokens_treino if len(t) > 10] + [t for t in tokens_teste if len(t) > 10]

print("\nPalavras longas:")
print(longas[:20])

estrangeiras = [t for t in tokens_treino if any(c in t for c in ['w', 'y', 'k'])] + [t for t in tokens_teste if any(c in t for c in ['w', 'y', 'k'])]

print("\nEstrangeirismos:")
print(estrangeiras[:10])

print("\nCobertura:", 1 - len(oov) / len(tokens_teste))

print("Hapax ratio:", len(raras) / num_palavras)

M1 = len(tokens_treino)
M2 = sum(freq**2 for freq in frequencia_treino.values())

yule_k = 10**4 * (M2 - M1) / (M1**2)
print("Yule's K:", yule_k)

frequencia.plot(30, cumulative=False)
plt.title("Frequência das palavras")
plt.xlabel("Palavras")
plt.ylabel("Frequência")
plt.show()