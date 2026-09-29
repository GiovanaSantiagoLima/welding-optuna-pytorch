# model.py
import torch
import torch.nn as nn

class MaterialNgramEncoder(nn.Module):
    def __init__(self, num_buckets, embed_dim, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(num_buckets, embed_dim, padding_idx=padding_idx)

    def forward(self, x):
        embedded = self.embedding(x)
        
        # Se x for [batch_size, seq_len], embedded será 3D: [batch_size, seq_len, emb_dim]
        # Nesse caso, aplicamos a média na dimensão da sequência (dim=1)
        if embedded.dim() == 3:
            return embedded.mean(dim=1)
        
        # Se x for apenas [batch_size], embedded já será 2D: [batch_size, emb_dim]
        # Retornamos diretamente, pois já está pronto para concatenar!
        return embedded

class InteracaoMateriais(nn.Module):
    """
    Representação explícita do par material base + material de adição:
        e_int = g([e_b, e_f, |e_b - e_f|, e_b ⊙ e_f])

    Os termos e_b e e_f isolados preservam os papéis (assimétricos) dos materiais;
    |e_b - e_f| e e_b ⊙ e_f (simétricos) capturam a proximidade entre os dois.
    Só faz sentido com o MESMO encoder para base e adição (mesmo espaço vetorial).

    usar_projecao=False -> g = identidade (a 1ª camada oculta da rede faz esse papel).
    """
    def __init__(self, emb_dim, out_dim=None, usar_projecao=True):
        super().__init__()
        if usar_projecao:
            self.out_dim = out_dim or emb_dim * 2
            self.g = nn.Sequential(
                nn.Linear(emb_dim * 4, self.out_dim),
                nn.ReLU(),
            )
        else:
            self.out_dim = emb_dim * 4
            self.g = nn.Identity()

    def forward(self, e_b, e_f):
        z = torch.cat([e_b, e_f, (e_b - e_f).abs(), e_b * e_f], dim=1)
        return self.g(z)


class RedeSoldagem(nn.Module):
    """
    Arquitetura da Rede Neural para previsão de parâmetros de soldagem (Regressão Multi-Output).

    Utiliza um MaterialNgramEncoder (embedding de n-gramas de caracteres com hashing + média)
    COMPARTILHADO para processar os códigos de material (base e adição), já que ambos seguem a
    mesma convenção de nomenclatura (norma + grade). Os dois vetores são combinados por um módulo
    de interação (ver InteracaoMateriais) e concatenados às demais features (contínuas e
    codificadas via One-Hot), que passam por camadas lineares densas (Fully Connected).
    A profundidade e largura da rede são parametrizáveis para facilitar a otimização
    de hiperparâmetros.
    """
    MODOS_INTERACAO = ("concat", "interacao", "interacao_g")

    def __init__(self, num_features_continuas: int, num_buckets: int, emb_dim: int = 16,
                 hidden_size: int = 64, num_layers: int = 2, dropout_rate: float = 0.2,
                 modo_interacao: str = "interacao_g"):
        """
        Inicializa as camadas e a estrutura da rede neural.

        Parâmetros
        ----------
        num_features_continuas: número de features numéricas/one-hot já existentes.
        num_buckets: número de buckets do hashing de n-gramas (NUM_BUCKETS em material.py),
                     o mesmo para material base e material de adição.
        emb_dim: dimensão do vetor gerado pelo MaterialNgramEncoder para cada material.
        modo_interacao: forma de combinar os dois materiais (útil para ablação):
            - "concat":      [e_b, e_f]
            - "interacao":   [e_b, e_f, |e_b - e_f|, e_b ⊙ e_f]   (g = identidade)
            - "interacao_g": g([e_b, e_f, |e_b - e_f|, e_b ⊙ e_f]) (g = Linear + ReLU)
        """
        super(RedeSoldagem, self).__init__()

        if modo_interacao not in self.MODOS_INTERACAO:
            raise ValueError(f"modo_interacao deve ser um de {self.MODOS_INTERACAO}")
        self.modo_interacao = modo_interacao

        # Um único encoder compartilhado entre material base e material de adição
        self.material_encoder = MaterialNgramEncoder(num_buckets=num_buckets, embed_dim=emb_dim)

        if modo_interacao == "concat":
            self.interacao = None
            dim_materiais = emb_dim * 2
        else:
            self.interacao = InteracaoMateriais(
                emb_dim, usar_projecao=(modo_interacao == "interacao_g")
            )
            dim_materiais = self.interacao.out_dim

        dimensao_total_entrada = num_features_continuas + dim_materiais

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
        x_material_base: tensor (batch, MAX_NGRAMS) com os ids de n-gramas do material base.
        x_material_add: tensor (batch, MAX_NGRAMS) com os ids de n-gramas do material de adição.

        Retorna
        -------
        torch.Tensor: Tensor de formato [batch_size, 3] contendo as previsões da rede.
        As saídas correspondem a (Voltagem, Amperagem, Velocidade de Soldagem).
        """
        e_b = self.material_encoder(x_material_base)   # (batch, emb_dim)
        e_f = self.material_encoder(x_material_add)    # (batch, emb_dim) - MESMO encoder

        if self.interacao is None:
            e_mat = torch.cat([e_b, e_f], dim=1)
        else:
            e_mat = self.interacao(e_b, e_f)

        x_combinado = torch.cat([x_num_cat, e_mat], dim=1)

        x_processado = self.camadas_ocultas(x_combinado)

        previsao = self.camada_saida(x_processado)

        return previsao