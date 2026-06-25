import torch
from torch.utils.data import Dataset, DataLoader

class DatasetSoldagem(Dataset):
    """
    Classe personalizada do PyTorch para carregar os dados de soldagem.
    Lida com múltiplas entradas (numéricas e embeddings) e múltiplas saídas.
    """
    def __init__(self, caminho_arquivo_pt: str, modo: str = 'train'):
        """
        Carrega os tensores e separa entre treino e teste.
        
        Args:
            caminho_arquivo_pt (str): Caminho para o arquivo .pt gerado no pré-processamento.
            modo (str): 'train' para carregar dados de treino, 'test' para teste.
        """
        # Carrega o dicionário de tensores que salvamos na etapa anterior
        dados = torch.load(caminho_arquivo_pt, weights_only=True)
        
        if modo == 'train':
            self.x_num_cat = dados['X_train_num']
            self.x_emb_base = dados['X_train_emb_base']
            self.x_emb_add = dados['X_train_emb_add']
            self.y = dados['y_train']
        elif modo == 'test':
            self.x_num_cat = dados['X_test_num']
            self.x_emb_base = dados['X_test_emb_base']
            self.x_emb_add = dados['X_test_emb_add']
            self.y = dados['y_test']
        else:
            raise ValueError("O argumento 'modo' deve ser 'train' ou 'test'.")

    def __len__(self):
        """Retorna o número total de amostras no dataset."""
        return len(self.y)

    def __getitem__(self, idx: int):
        """
        Pega a amostra na posição 'idx'.
        
        Returns:
            tupla: (tupla_de_entradas, alvos)
            - tupla_de_entradas: (features_numericas, id_material_base, id_material_adicao)
            - alvos: (voltagem, amperagem, velocidade)
        """
        entradas = (self.x_num_cat[idx], self.x_emb_base[idx], self.x_emb_add[idx])
        alvos = self.y[idx]
        
        return entradas, alvos

def criar_dataloaders(caminho_arquivo_pt: str, batch_size: int = 32):
    """
    Função auxiliar para criar os DataLoaders de treino e teste.
    """
    dataset_treino = DatasetSoldagem(caminho_arquivo_pt, modo='train')
    dataset_teste = DatasetSoldagem(caminho_arquivo_pt, modo='test')
    
    loader_treino = DataLoader(dataset_treino, batch_size=batch_size, shuffle=True, drop_last=True)
    
    loader_teste = DataLoader(dataset_teste, batch_size=batch_size, shuffle=False)
    
    return loader_treino, loader_teste

