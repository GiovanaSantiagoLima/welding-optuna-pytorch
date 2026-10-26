import re
import zlib

# ---------------------------------------------------------------------------
# Tokenização por n-gramas de caracteres com hashing trick (estilo fastText).
# Não há vocabulário fechado de n-gramas: qualquer n-grama, inclusive de
# materiais nunca vistos, cai em algum bucket.
# ---------------------------------------------------------------------------
NGRAM_SIZES = (1,2)   # tamanhos dos n-gramas
NUM_BUCKETS = 2048        # tamanho da "tabela hash" (ids de 1 a NUM_BUCKETS)
MAX_NGRAMS = 128          # comprimento fixo da sequência de n-gramas
PAD_IDX = 0               # índice de padding (buckets ocupam 1..NUM_BUCKETS)


def normalizar(material: str) -> str:
    """Maiúsculas e remove tudo que não for A-Z ou 0-9 (hífen, espaço etc.)."""
    return re.sub(r"[^A-Z0-9]", "", material.upper())


def char_ngrams(material: str, ns=NGRAM_SIZES) -> list:
    """
    Gera os n-gramas de caracteres do material, com marcadores de início '<'
    e fim '>'. Ex.: 'A106' -> ['<A','A1','10','06','6>','<A1','A10',...]
    """
    s = f"<{normalizar(material)}>"
    grams = []
    for n in ns:
        grams += [s[i:i + n] for i in range(len(s) - n + 1)]
    return grams


def encode(material: str) -> list:
    """
    Converte o código do material em uma lista de tamanho MAX_NGRAMS com os
    ids (buckets) dos n-gramas, preenchida com PAD_IDX.
    Usa zlib.crc32 (determinístico), e não hash(), que muda a cada execução.
    """
    ids = [1 + zlib.crc32(g.encode()) % NUM_BUCKETS for g in char_ngrams(material)]
    ids = ids[:MAX_NGRAMS]
    return ids + [PAD_IDX] * (MAX_NGRAMS - len(ids))


