import torch
import torch.nn as nn


class RedeSoldagem(nn.Module):
    """
    Arquitetura da Rede Neural para previsão de parâmetros de soldagem (Aprendizado Multitarefa - MTL).

    Baseline com codificação One-Hot para materiais: utiliza um tronco compartilhado
    (hard parameter sharing) seguido por três cabeças dedicadas (Voltagem, Amperagem
    e Velocidade de Soldagem), mantendo idêntica topologia neural aos modelos baseados
    em embeddings.
    """

    def __init__(
        self,
        num_features_continuas: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        camadas = []
        tamanho_entrada_atual = num_features_continuas

        for _ in range(num_layers):
            camadas.append(nn.Linear(tamanho_entrada_atual, hidden_size))
            camadas.append(nn.ReLU())
            camadas.append(nn.BatchNorm1d(hidden_size))
            camadas.append(nn.Dropout(dropout_rate))
            tamanho_entrada_atual = hidden_size

        self.camadas_ocultas = nn.Sequential(*camadas)

        # Cabeças dedicadas idênticas aos modelos mappings e tokenizacao (ex.: 256 -> 128 -> 1)
        head_size = hidden_size // 2
        self.head_voltagem = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
        )
        self.head_amperagem = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
        )
        self.head_velocidade = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
        )

    def forward(self, x_num_cat: torch.Tensor) -> torch.Tensor:
        """
        Retorna tensor [batch_size, 3] na ordem: (Voltagem, Amperagem, Velocidade).
        """
        x_processado = self.camadas_ocultas(x_num_cat)

        out_voltagem = self.head_voltagem(x_processado)
        out_amperagem = self.head_amperagem(x_processado)
        out_velocidade = self.head_velocidade(x_processado)

        return torch.cat([out_voltagem, out_amperagem, out_velocidade], dim=1)