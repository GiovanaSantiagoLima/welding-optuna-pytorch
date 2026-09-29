import re
import zlib

# ---------------------------------------------------------------------------
# Tokenização por n-gramas de caracteres com hashing trick (estilo fastText).
# Não há vocabulário fechado de n-gramas: qualquer n-grama, inclusive de
# materiais nunca vistos, cai em algum bucket.
# ---------------------------------------------------------------------------
NGRAM_SIZES = (2, 3, 4)   # tamanhos dos n-gramas
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


if __name__ == "__main__":
    import sys
    import os
    import torch

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from ml_model.model import MaterialNgramEncoder, InteracaoMateriais

    print("=" * 50)
    print("INFO DA TOKENIZAÇÃO")
    print("=" * 50)
    print("Tamanhos de n-grama:", NGRAM_SIZES)
    print("Buckets:", NUM_BUCKETS)
    print("Max n-gramas:", MAX_NGRAMS)

    print("\n" + "=" * 50)
    print("TESTE 1: n-gramas e encode de materiais conhecidos")
    print("=" * 50)
    for m in ["A312TP316L", "A106", "API5LX65"]:
        grams = char_ngrams(m)
        ids = encode(m)
        print(f"{m!r:14} -> {len(grams)} n-gramas | primeiros: {grams[:6]}")
        print(f"{'':14}    ids: {ids[:8]}...")
        assert len(ids) == MAX_NGRAMS
        assert all(0 <= i <= NUM_BUCKETS for i in ids)

    print("\n" + "=" * 50)
    print("TESTE 2: material NUNCA VISTO (sem vocabulário fechado, não quebra)")
    print("=" * 50)
    for m in ["A999TP888", "XYZ123", "B7777N99999"]:
        ids = encode(m)
        print(f"{m!r:14} -> {ids[:8]}...")
        assert len(ids) == MAX_NGRAMS

    print("\n" + "=" * 50)
    print("TESTE 3: normalização (símbolos são removidos)")
    print("=" * 50)
    assert encode("A312-TP316L") == encode("A312TP316L")
    assert encode("a312 tp316l") == encode("A312TP316L")
    print("'A312-TP316L' e 'a312 tp316l' geram o mesmo encoding de 'A312TP316L': OK")

    print("\n" + "=" * 50)
    print("TESTE 4: determinismo (mesmo material -> mesmos ids)")
    print("=" * 50)
    assert encode("A106") == encode("A106")
    print("OK")

    print("\n" + "=" * 50)
    print("TESTE 5: encoder + interação")
    print("=" * 50)
    torch.manual_seed(0)
    emb_dim = 16
    encoder = MaterialNgramEncoder(num_buckets=NUM_BUCKETS, embed_dim=emb_dim)
    interacao = InteracaoMateriais(emb_dim)

    pares = [
        ("A312TP316L", "A335P91"),   # par conhecido
        ("A999TP888", "XYZ123"),     # par nunca visto
    ]
    for base, adicao in pares:
        ids_base = torch.tensor([encode(base)])
        ids_adicao = torch.tensor([encode(adicao)])

        e_b = encoder(ids_base)
        e_f = encoder(ids_adicao)
        e_int = interacao(e_b, e_f)

        print(f"\nBase: {base!r} | Adição: {adicao!r}")
        print(f"  e_b shape:   {tuple(e_b.shape)}")
        print(f"  e_f shape:   {tuple(e_f.shape)}")
        print(f"  e_int shape: {tuple(e_int.shape)}")

        assert e_b.shape == (1, emb_dim)
        assert e_f.shape == (1, emb_dim)
        assert e_int.shape == (1, interacao.out_dim)

    # Material vazio -> só padding -> vetor de zeros (sem NaN)
    vazio = encoder(torch.tensor([encode("")]))
    assert torch.isfinite(vazio).all()
    print("\nMaterial vazio gera vetor finito (sem NaN): OK")

    print("\n✅ Todos os testes passaram — encoder de n-gramas funciona com materiais novos e conhecidos.")