import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from preprocessamento import carregar_e_processar_dados
import os
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. ARQUITETURA DA REDE NEURAL
class RedeNeural(nn.Module):
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

# 2. FUNÇÃO DE TREINAMENTO E PREDIÇÃO
def treinar_modelo_pytorch(X_train, y_train, params, input_dim, output_dim):
    hidden_dims = [int(x) for x in params["hidden_layers"].split(",")]
    
    model = RedeNeural(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_dims,
        activation_name=params["activation_name"],
        dropout_rate=params["dropout_rate"],
        use_batchnorm=params["use_batchnorm"]
    ).to(device)

    # Alterado para MSELoss conforme solicitado
    criterion = nn.MSELoss()
    
    if params["optimizer_name"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(params["batch_size"]), shuffle=True)

    model.train()
    for epoch in range(int(params["num_epochs"])):
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
                
    return model

def prever_modelo_pytorch(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    return preds

# 3. EXPORTAÇÃO PARA ONNX
def exportar_para_onnx(pytorch_model, input_dim, nome_arquivo="melhor_modelo.onnx"):
    print("Convertendo o melhor modelo final para ONNX...")
    pytorch_model.eval()
    dummy_input = torch.randn(1, input_dim, device=device)
    
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        nome_arquivo,
        export_params=True,        
        opset_version=18,  
        do_constant_folding=True,  
        input_names=['input'],     
        output_names=['output'],   
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} 
    )
    print(f"Melhor modelo ONNX salvo em: '{os.path.abspath(nome_arquivo)}'")
    return nome_arquivo

# 4. FUNÇÃO PRINCIPAL COM OPTUNA
def train():
    print(f"Dispositivo detectado: {device}")

    X_df, y_df = carregar_e_processar_dados()
    X = X_df.values if isinstance(X_df, pd.DataFrame) else np.array(X_df)
    y = y_df.values if isinstance(y_df, pd.DataFrame) else np.array(y_df)

    print("Normalizando dados...")
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    input_dim, output_dim = X_scaled.shape[1], y_scaled.shape[1]
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout_rate": trial.suggest_categorical("dropout_rate", [0.0]),
            "use_batchnorm": trial.suggest_categorical("use_batchnorm", [True, False]),
            "activation_name": trial.suggest_categorical("activation_name", ["relu", "gelu", "silu", "mish"]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "hidden_layers": trial.suggest_categorical("hidden_layers", ["128,64", "256,128", "128,128,64", "256,256,128"]),
            "num_epochs": trial.suggest_categorical("num_epochs", [50, 100, 150]),
            "optimizer_name": trial.suggest_categorical("optimizer_name", ["adam", "adamw"]),
        }

        fold_losses = []
        for train_idx, val_idx in cv_strategy.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y_scaled[train_idx], y_scaled[val_idx]

            model = treinar_modelo_pytorch(X_tr, y_tr, params, input_dim, output_dim)
            preds = prever_modelo_pytorch(model, X_val)
            
            # Critério de avaliação alterado para MSE
            r2_fold = r2_score(y_val, preds, multioutput="uniform_average")
            fold_losses.append(r2_fold)

        return np.mean(fold_losses)

    def callback_print(study, trial):
        print(f"[Tentativa {trial.number + 1}/20] finalizada!")
        print(f"  MSE Medio de Validacao: {trial.value:.6f}")
        print(f"  Melhor MSE Geral ate agora: {study.best_value:.6f}")
        print("-" * 50)

    print("Iniciando Otimizacao com Optuna (5-Fold CV baseada em MSE)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, callbacks=[callback_print])
    print("Otimizacao concluida!")

    print("\nRetreinando com a melhor configuracao na base completa...")
    melhores_parametros = study.best_params
    melhor_modelo_final = treinar_modelo_pytorch(X_scaled, y_scaled, melhores_parametros, input_dim, output_dim)

    # Predições finais convertidas de volta à escala original
    y_pred_scaled = prever_modelo_pytorch(melhor_modelo_final, X_scaled)
    y_real = scaler_y.inverse_transform(y_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Exibição das 3 métricas por variável física real
    nomes_parametros = ["Voltagem (V)", "Amperagem (A)", "Velocidade de Soldagem"]
    print("\n" + "="*50)
    print("METRICAS REAIS POR VARIAVEL (APOS RETREINAMENTO)")
    print("="*50)
    for i, nome in enumerate(nomes_parametros):
        y_real_iso = y_real[:, i]
        y_pred_iso = y_pred[:, i]
        print(f"Parâmetro: {nome}")
        print(f"  MAE: {mean_absolute_error(y_real_iso, y_pred_iso):.4f}")
        print(f"  MSE: {mean_squared_error(y_real_iso, y_pred_iso):.4f}")
        print(f"  R2 : {r2_score(y_real_iso, y_pred_iso):.4f}\n")
    
    # Métricas Gerais Unificadas
    mae_geral = mean_absolute_error(y_real, y_pred)
    mse_geral = mean_squared_error(y_real, y_pred)
    r2_geral = r2_score(y_real, y_pred)

    print("="*50)
    print("METRICAS GERAIS DA SAIDA (ESCALA REAL)")
    print(f"  MAE Geral: {mae_geral:.4f}")
    print(f"  MSE Geral: {mse_geral:.4f}")
    print(f"  R2 Geral:  {r2_geral:.4f}")
    print("="*50)

    metricas_gerais = {
        "mae_real_geral": mae_geral,
        "mse_real_geral": mse_geral,
        "r2_real_geral": r2_geral,
        "melhores_parametros": melhores_parametros
    }

    caminho_onnx = exportar_para_onnx(melhor_modelo_final, input_dim=input_dim)
    return caminho_onnx, metricas_gerais


if __name__ == "__main__":
    caminho_modelo_onnx, minhas_metricas = train()
    print("\nRetorno Final da Funcao:")
    print(f"Caminho ONNX: {caminho_modelo_onnx}")
    print(f"Dados: {minhas_metricas}")