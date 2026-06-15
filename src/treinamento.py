import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pre_processamento import limpeza 

optuna.logging.set_verbosity(optuna.logging.WARNING)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. ARQUITETURA DA REDE NEURAL (SIMPLIFICADA E MULTI-OUTPUT)
# =====================================================================
class AvançadoMLP(nn.Module):
    """
    Uma rede MLP pura e robusta. Deixando o tratamento de categóricos 
    para o pré-processamento, a rede processa a matriz final de forma direta e rápida.
    """
    def __init__(self, input_dim, output_dim, hidden_layers, activation_name, dropout_rate, use_batchnorm):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        activations = {
            "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(), "mish": nn.Mish()
        }
        activation = activations[activation_name]

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# =====================================================================
# 2. ROTINAS AUXILIARES DE TREINO E PREDIÇÃO
# =====================================================================
def treinar_modelo_pytorch(X_train, y_train, X_val, y_val, params, input_dim, output_dim, trial=None):
    hidden_dims = [int(x) for x in params["hidden_layers"].split(",")]
    
    model = AvançadoMLP(
        input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_dims,
        activation_name=params["activation_name"], dropout_rate=params["dropout_rate"],
        use_batchnorm=params["use_batchnorm"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]) \
                if params["optimizer_name"] == "adamw" else \
                optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    # Conversão eficiente para Tensores
    X_tr, y_tr = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_v, y_v = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(params["batch_size"]), shuffle=True)

    for epoch in range(int(params["num_epochs"])):
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        
        # Poda do Optuna (Pruning) para economizar tempo em treinos ruins
        if trial is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_v), y_v).item()
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
    return model

def prever_modelo_pytorch(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    return preds

# =====================================================================
# 3. EXPORTAÇÃO EXCELENTE PARA ONNX
# =====================================================================
def exportar_para_onnx(pytorch_model, input_dim, nome_arquivo="melhor_modelo.onnx"):
    pytorch_model.eval()
    dummy_input = torch.randn(1, input_dim, device=device)
    torch.onnx.export(
        pytorch_model, dummy_input, nome_arquivo,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    return nome_arquivo

# =====================================================================
# 4. PIPELINE PRINCIPAL (TRAIN)
# =====================================================================
def train(df_limpo):
    print(f"Dispositivo de processamento: {device}")
    
    # ⚠️ Nota: Idealmente, a função preparar_dados_com_mapping deve aplicar OneHotEncoder ou OrdinalEncoder
    X_train_final, X_test_final, y_train, y_test, _ = preparar_dados(df_limpo)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    input_dim = X_train_final.shape[1]
    output_dim = y_train_scaled.shape[1]
    
    # Divisão simples de Treino/Validação para o Optuna (Rápido e Eficiente)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_final, y_train_scaled, test_size=0.15, random_state=42)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout_rate": trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2]),
            "use_batchnorm": trial.suggest_categorical("use_batchnorm", [True, False]),
            "activation_name": trial.suggest_categorical("activation_name", ["relu", "gelu", "silu", "mish"]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "hidden_layers": trial.suggest_categorical("hidden_layers", ["128,64", "256,128", "128,128,64"]),
            "num_epochs": trial.suggest_categorical("num_epochs", [50, 100]),
            "optimizer_name": trial.suggest_categorical("optimizer_name", ["adam", "adamw"]),
        }

        try:
            model = treinar_modelo_pytorch(X_tr, y_tr, X_val, y_val, params, input_dim, output_dim, trial=trial)
            preds = prever_modelo_pytorch(model, X_val)
            return mean_squared_error(y_val, preds)
        except optuna.TrialPruned:
            raise

    print("Iniciando Otimização com Optuna (Validação com Pruning ativo)...")
    # O pruner MedianPruner corta experimentos que começam apresentando resultados ruins
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)
    print(f"Otimização concluída! Melhor MSE de Validação: {study.best_value:.6f}")

    print("\nRetreinando modelo final com os melhores parâmetros...")
    melhores_parametros = study.best_params
    melhor_modelo_final = treinar_modelo_pytorch(X_train_final, y_train_scaled, X_val, y_val, melhores_parametros, input_dim, output_dim)

    # Avaliação final no conjunto de Teste isolado
    y_pred_scaled = prever_modelo_pytorch(melhor_modelo_final, X_test_final)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    nomes_parametros = ["Voltagem (V)", "Amperagem (A)", "Velocidade de Soldagem"]
    print("\n" + "="*60)
    print("MÉTRICAS REAIS POR VARIÁVEL (BASE DE TESTE)")
    print("="*60)
    for i, nome in enumerate(nomes_parametros):
        print(f"Parâmetro: {nome}")
        print(f"  MAE: {mean_absolute_error(y_test[:, i], y_pred[:, i]):.4f}")
        print(f"  MSE: {mean_squared_error(y_test[:, i], y_pred[:, i]):.4f}")
        print(f"  R2 : {r2_score(y_test[:, i], y_pred[:, i]):.4f}\n")
    
    caminho_onnx = exportar_para_onnx(melhor_modelo_final, input_dim=input_dim)
    return caminho_onnx, melhores_parametros

# =====================================================================
# 5. BLOCO DE EXECUÇÃO
# =====================================================================
if __name__ == "__main__":
    caminho_dos_dados = "C:\\Users\\Workstation\\Documents\\GitHub\\welding-optuna-pytorch\\data\\all_data.csv"

    if not os.path.exists(caminho_dos_dados):
        raise FileNotFoundError(f"Não encontrei o arquivo no caminho: '{caminho_dos_dados}'")

    class BaseDadosReal:
        def __init__(self, caminho):
            self.caminho = caminho
        def obter_dados_limpos(self):
            df_bruto = pd.read_csv(self.caminho)
            return limpeza(df_bruto)

    adaptador = BaseDadosReal(caminho_dos_dados)
    df_pronto = adaptador.obter_dados_limpos() # Correção: Extraindo o dataframe limpo antes de treinar
    
    print("\n" + "="*60)
    print("INICIANDO MODELAGEM PREDITIVA COM DADOS REAIS DE SOLDAGEM")
    print("="*60)
    
    caminho_modelo_onnx, melhores_configuracoes = train(df_pronto)
    
    print("\n" + "="*60)
    print(f"Modelo de produção exportado para: {caminho_modelo_onnx}")
    print("="*60)