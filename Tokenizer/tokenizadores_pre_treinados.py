from transformers import AutoTokenizer
from carregar_corpus import carregar_corpus

corpus_teste = carregar_corpus("corpus/teste")

# BPE (GPT-2)
tokenizer_bpe = AutoTokenizer.from_pretrained("gpt2")
tokens_bpe = [tokenizer_bpe.tokenize(texto) for texto in corpus_teste]

# WordPiece (BERT)
tokenizer_wp = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens_wp = [tokenizer_wp.tokenize(texto) for texto in corpus_teste]

# exemplo
for texto in corpus_teste:
    print("\n--- TEXTO ---")
    print(texto[:200])
    
    print("\nBPE tokens:")
    print(tokens_bpe[corpus_teste.index(texto)][:30])

    print("\nWordPiece tokens:")
    print(tokens_wp[corpus_teste.index(texto)][:30])