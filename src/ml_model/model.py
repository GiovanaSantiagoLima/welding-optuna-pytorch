import torch
import torch.nn as nn

class RedeSoldagem(nn.Module):
    """
    Arquitetura da Rede Neural para previsão de parâmetros de soldagem (Regressão Multi-Output).
    
    Utiliza camadas de Embedding para processar variáveis categóricas de alta
    cardinalidade (materiais) e camadas lineares densas (Fully Connected) para as
    features contínuas e codificadas via One-Hot. A profundidade e largura da rede
    são parametrizáveis para facilitar a otimização de hiperparâmetros.
    """
    def __init__(self, num_features_continuas: int,
                 hidden_size: int = 64,num_layers: int = 2,dropout_rate: float = 0.2):
        """
        Inicializa as camadas e a estrutura da rede neural.
        """
        super(RedeSoldagem, self).__init__()

        dimensao_total_entrada = num_features_continuas 
        
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

    def forward(self, x_num_cat: torch.Tensor) -> torch.Tensor:
        """
        Define o fluxo de passagem direta (forward pass) dos dados pela rede.

        Retorna
        -------
        torch.Tensor:Tensor de formato [batch_size, 3] contendo as previsões da rede.As saídas correspondem a (Voltagem, Amperagem, Velocidade de Soldagem).
        """

        x_combinado = torch.cat([x_num_cat], dim=1)
        
        x_processado = self.camadas_ocultas(x_combinado)
        
        previsao = self.camada_saida(x_processado)

        return previsao
