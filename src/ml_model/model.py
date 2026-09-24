# model.py
import torch
import torch.nn as nn


class MaterialEncoder(nn.Module):
    """
    Transforma uma sequência de índices de caracteres (código do material)
    em um único vetor denso de tamanho fixo, via embedding + média (ignorando padding).
    """
    def __init__(self, vocab_size, embed_dim=16, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (batch,) -> (batch, 1): trata como sequência de 1 caractere
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1)
        emb = emb * mask
        soma = emb.sum(dim=1)
        contagem = mask.sum(dim=1).clamp(min=1)
        return soma / contagem

class RedeSoldagem(nn.Module):
    """
    Arquitetura da Rede Neural para previsão de parâmetros de soldagem (Regressão Multi-Output).

    Utiliza um MaterialEncoder (embedding de caracteres + média) COMPARTILHADO para
    processar os códigos de material (base e adição), já que ambos seguem a mesma
    convenção de nomenclatura (norma + grade). As demais features (contínuas e
    codificadas via One-Hot) passam por camadas lineares densas (Fully Connected).
    A profundidade e largura da rede são parametrizáveis para facilitar a otimização
    de hiperparâmetros.
    """
    def __init__(self, num_features_continuas: int, vocab_size: int, emb_dim: int = 16,
                 hidden_size: int = 64, num_layers: int = 2, dropout_rate: float = 0.2):
        """
        Inicializa as camadas e a estrutura da rede neural.

        Parâmetros
        ----------
        num_features_continuas: número de features numéricas/one-hot já existentes.
        vocab_size: tamanho do vocabulário de CARACTERES (ex: len(VOCAB) em material.py),
                    o mesmo para material base e material de adição.
        emb_dim: dimensão do vetor gerado pelo MaterialEncoder para cada material.
        """
        super(RedeSoldagem, self).__init__()

        # Um único encoder compartilhado entre material base e material de adição
        self.material_encoder = MaterialEncoder(vocab_size=vocab_size, embed_dim=emb_dim)

        dimensao_total_entrada = num_features_continuas + (emb_dim * 2)  # base + adição

        camadas = []
        tamanho_entrada_atual = dimensao_total_entrada

        for i in range(num_layers):
            camadas.append(nn.Linear(tamanho_entrada_atual, hidden_size))
            camadas.append(nn.ReLU())
            camadas.append(nn.BatchNorm1d(hidden_size))
            camadas.append(nn.Dropout(dropout_rate))

            tamanho_entrada_atual = hidden_size

        self.camadas_ocultas = nn.Sequential(*camadas)

        self.camada_saida = nn.Linear(hidden_size, 3)

    def forward(self, x_num_cat: torch.Tensor, x_material_base: torch.Tensor,
                x_material_add: torch.Tensor) -> torch.Tensor:
        """
        Define o fluxo de passagem direta (forward pass) dos dados pela rede.

        Parâmetros
        ----------
        x_num_cat: tensor (batch, num_features_continuas) com as demais features.
        x_material_base: tensor (batch, MAX_LEN) com os índices de caracteres do material base.
        x_material_add: tensor (batch, MAX_LEN) com os índices de caracteres do material de adição.

        Retorna
        -------
        torch.Tensor: Tensor de formato [batch_size, 3] contendo as previsões da rede.
        As saídas correspondem a (Voltagem, Amperagem, Velocidade de Soldagem).
        """
        vec_base = self.material_encoder(x_material_base)   # (batch, emb_dim)
        vec_add = self.material_encoder(x_material_add)      # (batch, emb_dim) - MESMO encoder

        x_combinado = torch.cat([x_num_cat, vec_base, vec_add], dim=1)

        x_processado = self.camadas_ocultas(x_combinado)

        previsao = self.camada_saida(x_processado)

        return previsao


