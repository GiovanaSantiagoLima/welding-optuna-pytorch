import string

PAD_TOKEN = "[PAD]" #preenchimento
UNK_TOKEN = "[UNK]" #

CHARS = list(string.ascii_uppercase) + list(string.digits)  # A-Z + 0-9
VOCAB = [PAD_TOKEN, UNK_TOKEN] + CHARS
CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}

MAX_LEN = 30  

def encode(material: str) -> list:
    material = material.upper().strip()  # normaliza
    ids = [CHAR2IDX.get(ch, CHAR2IDX[UNK_TOKEN]) for ch in material]
    ids = ids + [CHAR2IDX[PAD_TOKEN]] * (MAX_LEN - len(ids))
    return ids[:MAX_LEN]

if __name__ == "__main__":
    import sys
    import os
    import torch

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from ml_model.model import MaterialEncoder

    print("=" * 50)
    print("INFO DO VOCABULÁRIO")
    print("=" * 50)
    print("Vocabulário:", VOCAB)
    print("Tamanho do vocabulário:", len(VOCAB))
    print("Max len:", MAX_LEN)

    print("\n" + "=" * 50)
    print("TESTE 1: Encode de materiais conhecidos")
    print("=" * 50)
    materiais_conhecidos = ["A312TP316L", "A106", "API5LX65"]
    for m in materiais_conhecidos:
        ids = encode(m)
        print(f"{m!r:20} -> {ids}")

    print("\n" + "=" * 50)
    print("TESTE 2: Encode de material NUNCA VISTO antes")
    print("=" * 50)
    materiais_novos = ["A999TP888", "XYZ123", "B7777N99999"]
    for m in materiais_novos:
        ids = encode(m)
        print(f"{m!r:20} -> {ids}")
        assert len(ids) == MAX_LEN, "Erro: tamanho da sequência não bate com MAX_LEN"

    print("\n" + "=" * 50)
    print("TESTE 3: Caractere fora do vocabulário (deve virar [UNK])")
    print("=" * 50)
    material_com_simbolo = "A312-TP316L"  # hífen não está no vocabulário
    ids = encode(material_com_simbolo)
    unk_idx = CHAR2IDX["[UNK]"]
    pos_hifen = material_com_simbolo.upper().index("-")
    print(f"{material_com_simbolo!r:20} -> {ids}")
    print(f"Índice de [UNK] é {unk_idx}, aparece na posição do '-': "
          f"{ids[pos_hifen] == unk_idx}")

    print("\n" + "=" * 50)
    print("TESTE 4: MaterialEncoder gerando vetores")
    print("=" * 50)
    encoder = MaterialEncoder(vocab_size=len(VOCAB), embed_dim=16)

    pares = [
        ("A312TP316L", "A335P91"),   # par conhecido
        ("A999TP888", "XYZ123"),     # par nunca visto
    ]

    for base, adicao in pares:
        ids_base = torch.tensor([encode(base)])
        ids_adicao = torch.tensor([encode(adicao)])

        vec_base = encoder(ids_base)
        vec_adicao = encoder(ids_adicao)

        print(f"\nBase: {base!r} | Adição: {adicao!r}")
        print(f"  vec_base shape:   {vec_base.shape}")
        print(f"  vec_adicao shape: {vec_adicao.shape}")
        print(f"  vec_base[:5]:   {vec_base[0][:5]}")
        print(f"  vec_adicao[:5]: {vec_adicao[0][:5]}")

        assert vec_base.shape == (1, 16), "Erro: shape do vetor base incorreto"
        assert vec_adicao.shape == (1, 16), "Erro: shape do vetor adição incorreto"

    print("\n✅ Todos os testes passaram — encoder funciona com materiais novos e conhecidos.")