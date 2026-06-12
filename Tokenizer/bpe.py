from carregar_corpus import carregar_corpus
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

corpus_treino = carregar_corpus("corpus/treino")
# tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

# Após a criação do tokenizer, pode ser usada a linha acima para carregar o tokenizer salvo, ou seja, para reutilizar o tokenizer sem precisar treiná-lo novamente.
def create_tokenizer(corpus):
    tokenizer = Tokenizer(models.BPE())

    trainer = trainers.BpeTrainer(
        vocab_size=1500,
        special_tokens=["<unk>"]
    )

    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.save("bpe_tokenizer.json")

    return tokenizer

tokenizer = create_tokenizer(corpus_treino)

vocab_ordenado = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])

print("Tamanho do vocabulário:", len(tokenizer.get_vocab()))
print("\nVocabulario ordenado por id:")
print(vocab_ordenado[:20])  # Mostra os primeiros 20 tokens do vocabulário ordenados por id
print("\nExemplo do vocabulário:")
print(list(tokenizer.get_vocab().items())[:20])  # Mostra os primeiros 20 tokens do vocabulário


# Testando o tokenizer com o corpus de teste
corpus_teste = carregar_corpus("corpus/teste")

for texto in corpus_teste:
    output = tokenizer.encode(texto)

    print("\n--- TEXTO ---")
    print(texto[:200])

    print("\nTokens:")
    print(output.tokens[:22])  # Mostra os primeiros 22 tokens

    print("\nTokens ids:")
    print(output.ids[:22])  # Mostra os primeiros 22 ids de tokens
