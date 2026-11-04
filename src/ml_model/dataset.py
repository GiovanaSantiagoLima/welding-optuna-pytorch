import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
class DatasetSoldagemCV(Dataset):
    
    def __init__(self, caminho_arquivo_pt: str):
        dados = torch.load(caminho_arquivo_pt, weights_only=True)
        self.x_num_cat = dados['X_train_num']
        self.x_emb_base = dados['X_train_emb_base']
        self.x_emb_add = dados['X_train_emb_add']
        self.y = dados['y_train']
        
    def __len__(self): 
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x_num_cat[idx], self.x_emb_base[idx], self.x_emb_add[idx]), self.y[idx]

def data_loader_cross_validation(caminho_arquivo_pt: str, batch_size: int = 32, k_folds: int = 5):
    dataset = DatasetSoldagemCV(caminho_arquivo_pt)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    dataloaders_folds = []
    for train_ids, val_ids in kfold.split(dataset):
        loader_treino = DataLoader(Subset(dataset, train_ids), batch_size=batch_size, shuffle=True, drop_last=True)
        loader_val = DataLoader(Subset(dataset, val_ids), batch_size=batch_size, shuffle=False)
        dataloaders_folds.append((loader_treino, loader_val))
    return dataloaders_folds