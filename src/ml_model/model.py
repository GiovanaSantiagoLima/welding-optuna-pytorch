# model.py
import torch
import torch.nn as nn

class MaterialNgramEncoder(nn.Module):
    """
    Embedding de n-gramas (hashing) + média ignorando padding.
    Entrada esperada: (batch, MAX_NGRAMS) com ids inteiros (0 = padding).
    Se chegar (batch,), é tratado como (batch, 1) para não quebrar.
    Saída: (batch, embed_dim)
    """
    def __init__(self, num_buckets: int, embed_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(num_buckets + 1, embed_dim, padding_idx=pad_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)                      
        x = x.long()
        emb = self.embedding(x)                     
        mask = (x != 0).unsqueeze(-1).to(emb.dtype)  
        soma = (emb * mask).sum(dim=1) 
        return soma / mask.sum(dim=1).clamp(min=1)  


def _ativacao(nome: str) -> nn.Module:
    nome = nome.lower()
    if nome == "relu":
        return nn.ReLU()
    if nome == "leakyrelu" or nome == "leaky_relu":
        return nn.LeakyReLU()
    if nome == "gelu":
        return nn.GELU()
    if nome == "tanh":
        return nn.Tanh()
    if nome == "silu":
        return nn.SiLU()
    raise ValueError(f"Ativação desconhecida: {nome}")


class RedeSoldagem(nn.Module):
    """
    Rede para previsão de parâmetros de soldagem (regressão multi-output).

    Utiliza arquitetura Multi-Head: um tronco compartilhado processa as 
    features contínuas e embeddings (n-gramas), dividindo-se no final em 
    três redes independentes focadas em Voltagem, Amperagem e Velocidade.
    """

    def __init__(self, num_features_continuas: int, num_buckets: int, emb_dim: int = 16,
                 hidden_size: int = 64, num_layers: int = 2, dropout_rate: float = 0.2,
                 modo_interacao: str = "concat", activation: str = "relu"):
        super().__init__()

        if modo_interacao != "concat":
            raise ValueError("Apenas o modo 'concat' é suportado.")

        self.material_encoder = MaterialNgramEncoder(num_buckets=num_buckets, embed_dim=emb_dim)
        tamanho_entrada = num_features_continuas + (emb_dim * 2)

        # Tronco Compartilhado
        camadas = []
        for _ in range(num_layers):
            camadas.append(nn.Linear(tamanho_entrada, hidden_size))
            camadas.append(_ativacao(activation))
            camadas.append(nn.BatchNorm1d(hidden_size))
            camadas.append(nn.Dropout(dropout_rate))
            tamanho_entrada = hidden_size

        self.camadas_ocultas = nn.Sequential(*camadas)

        # Cabeças Independentes (Multi-Head)
        dimensao_head = max(hidden_size // 2, 8)
        
        self.head_voltagem = nn.Sequential(
            nn.Linear(hidden_size, dimensao_head),
            _ativacao(activation),
            nn.Linear(dimensao_head, 1)
        )
        
        self.head_amperagem = nn.Sequential(
            nn.Linear(hidden_size, dimensao_head),
            _ativacao(activation),
            nn.Linear(dimensao_head, 1)
        )
        
        self.head_velocidade = nn.Sequential(
            nn.Linear(hidden_size, dimensao_head),
            _ativacao(activation),
            nn.Linear(dimensao_head, 1)
        )

    def forward(self, x_num_cat: torch.Tensor,  
                x_material_base: torch.Tensor, x_material_add: torch.Tensor) -> torch.Tensor:
        """
        Retorna as predições concatenadas: (batch, 3) -> [Voltagem, Amperagem, Velocidade]
        """
        e_b = self.material_encoder(x_material_base)
        e_f = self.material_encoder(x_material_add)

        x = torch.cat([x_num_cat, e_b, e_f], dim=1)
        x_compartilhado = self.camadas_ocultas(x)
        
        out_v = self.head_voltagem(x_compartilhado)
        out_a = self.head_amperagem(x_compartilhado)
        out_vel = self.head_velocidade(x_compartilhado)
        
        return torch.cat([out_v, out_a, out_vel], dim=1)