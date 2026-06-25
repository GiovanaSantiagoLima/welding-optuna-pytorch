import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import criar_dataloaders
from model import RedeSoldagem

OUTPUT_NAMES = ["voltagem", "amperagem", "velocidade"]


def calcular_metricas(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Calcula MSE, MAE e R² global e por output (voltagem, amperagem, velocidade).

    preds / targets : (N, 3)
    Retorna dict com chaves:
        mse, mae, r2                          ← média global
        mse_voltagem, mae_voltagem, r2_voltagem
        mse_amperagem, mae_amperagem, r2_amperagem
        mse_velocidade, mae_velocidade, r2_velocidade
    """
    result = {}

    for i, nome in enumerate(OUTPUT_NAMES):
        p = preds[:, i]
        t = targets[:, i]
        mse = float(nn.functional.mse_loss(p, t).item())
        mae = float(nn.functional.l1_loss(p, t).item())
        ss_res = float(((t - p) ** 2).sum().item())
        ss_tot = float(((t - t.mean()) ** 2).sum().item())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        result[f"mse_{nome}"] = mse
        result[f"mae_{nome}"] = mae
        result[f"r2_{nome}"]  = r2

    # — Global (média dos 3 outputs) —
    result["mse"] = float(np.mean([result[f"mse_{n}"] for n in OUTPUT_NAMES]))
    result["mae"] = float(np.mean([result[f"mae_{n}"] for n in OUTPUT_NAMES]))
    result["r2"]  = float(np.mean([result[f"r2_{n}"]  for n in OUTPUT_NAMES]))

    return result


def imprimir_metricas(metricas: dict, prefixo: str = "") -> None:
    """Imprime um bloco formatado com todas as métricas."""
    sep = "─" * 54
    print(f"\n{prefixo}{sep}")
    print(f"{prefixo}{'':>20}  {'MSE':>8}  {'MAE':>8}  {'R²':>7}")
    print(f"{prefixo}{sep}")
    for nome in OUTPUT_NAMES:
        print(
            f"{prefixo}  {nome:<18}"
            f"  {metricas[f'mse_{nome}']:>8.4f}"
            f"  {metricas[f'mae_{nome}']:>8.4f}"
            f"  {metricas[f'r2_{nome}']:>7.4f}"
        )
    print(f"{prefixo}{sep}")
    print(
        f"{prefixo}  {'GERAL':<18}"
        f"  {metricas['mse']:>8.4f}"
        f"  {metricas['mae']:>8.4f}"
        f"  {metricas['r2']:>7.4f}"
    )
    print(f"{prefixo}{sep}\n")


def treinar(params: dict, data_path: str, epochs: int = 100, patience: int = 10, device: str | None = None, onnx_path: str = "melhor_modelo.onnx") -> dict: 
    """
    Treina o RedeSoldagem com os hiperparâmetros fornecidos e exporta para ONNX.

    Parâmetros
    ----------
    params : dict
        Hiperparâmetros do modelo e do otimizador. Esperados:
            - emb_dim           (int)
            - hidden_size       (int)
            - num_layers        (int)
            - dropout_rate      (float)
            - learning_rate     (float)
            - batch_size        (int)
            - weight_decay      (float)   ← opcional, default 1e-4
            - optimizer         (str)     ← 'Adam' | 'AdamW' | 'SGD' | 'RMSprop', default 'AdamW'
            - activation        (str)     ← 'relu' | 'gelu' | 'silu', default 'relu'
            - criterion         (str)     ← 'MSELoss' | 'L1Loss' | 'HuberLoss', default 'MSELoss'

    data_path : str
        Caminho para o arquivo .pt com os tensores pré-processados.
    epochs : int
        Número máximo de épocas.
    patience : int
        Épocas sem melhora antes de parar (early stopping).
    device : str | None
        'cuda', 'cpu', ou None para detecção automática.

    Retorna
    -------
    dict com métricas globais e por output:
        best_val_mse, best_val_mae, best_val_r2
        best_val_mse_voltagem,  best_val_mae_voltagem,  best_val_r2_voltagem
        best_val_mse_amperagem, best_val_mae_amperagem, best_val_r2_amperagem
        best_val_mse_velocidade,best_val_mae_velocidade,best_val_r2_velocidade
        stopped_epoch, hist_train, hist_val
    """

    # Dispositivo
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"[train] Dispositivo: {dev}")

    # DataLoaders
    batch_size = params.get("batch_size", 32)
    loader_treino, loader_val = criar_dataloaders(data_path, batch_size=batch_size)
    (x_num, x_base, x_add), _ = next(iter(loader_treino))
    num_features = x_num.shape[1]
    dados = torch.load(data_path, weights_only=True)
    vocab_base = int(dados["X_train_emb_base"].max().item() + 1)
    vocab_add  = int(dados["X_train_emb_add"].max().item() + 1)

    # Modelo
    modelo = RedeSoldagem(
        num_features_continuas=num_features,
        vocab_base_size=vocab_base,
        vocab_add_size=vocab_add,
        emb_dim=params["emb_dim"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout_rate=params["dropout_rate"],
    ).to(dev)

    # Otimizador
    opt_name = params.get("optimizer", "AdamW")
    if opt_name == "AdamW":
        optimizer = optim.AdamW(modelo.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    elif opt_name == "Adam":
        optimizer = optim.Adam(modelo.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    elif opt_name == "SGD":
        optimizer = optim.SGD(modelo.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(modelo.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(5, patience // 4))

    # Critério
    criterion_name = params.get("criterion", "MSELoss")
    if criterion_name == "MSELoss":
        criterion = nn.MSELoss()
    elif criterion_name == "HuberLoss":
        criterion = nn.HuberLoss()
    elif criterion_name == "L1Loss":
        criterion = nn.L1Loss()

    # Loop de treino
    hist_train: list[float] = []
    hist_val:   list[float] = []

    best_val_mse       = float("inf")
    best_metricas      = None
    epochs_sem_melhora = 0
    stopped_epoch      = epochs

    for epoca in range(1, epochs + 1):
        # — Treino —
        modelo.train()
        losses_batch = []
        for (x_num, x_base, x_add), y_batch in loader_treino:
            x_num   = x_num.to(dev)
            x_base  = x_base.to(dev)
            x_add   = x_add.to(dev)
            y_batch = y_batch.to(dev)

            optimizer.zero_grad()
            pred = modelo(x_num, x_base, x_add)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()
            losses_batch.append(loss.item())

        train_loss = float(np.mean(losses_batch))
        hist_train.append(train_loss)

        # — Validação —
        modelo.eval()
        all_preds   = []
        all_targets = []
        with torch.no_grad():
            for (x_num, x_base, x_add), y_batch in loader_val:
                x_num   = x_num.to(dev)
                x_base  = x_base.to(dev)
                x_add   = x_add.to(dev)
                y_batch = y_batch.to(dev)
                all_preds.append(modelo(x_num, x_base, x_add))
                all_targets.append(y_batch)

        all_preds   = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metricas    = calcular_metricas(all_preds, all_targets)

        hist_val.append(metricas["mse"])
        scheduler.step(metricas["mse"])

        # — Log a cada 10 épocas —
        if epoca % 10 == 0:
            lr_atual = optimizer.param_groups[0]["lr"]
            print(
                f"  Época {epoca:>4}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val MSE: {metricas['mse']:.4f} | "
                f"Val MAE: {metricas['mae']:.4f} | "
                f"Val R²: {metricas['r2']:.4f} | "
                f"LR: {lr_atual:.2e}"
            )

        # — Early stopping (baseado no MSE global de validação) —
        if metricas["mse"] < best_val_mse:
            best_val_mse  = metricas["mse"]
            best_metricas = metricas
            best_model_weights = copy.deepcopy(modelo.state_dict()) 
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= patience:
                print(f"  Early stopping na época {epoca}.")
                stopped_epoch = epoca
                break

    # — Relatório final —
    print(f"\n{'=' * 54}")
    print(f"  MÉTRICAS FINAIS DE VALIDAÇÃO (melhor época)")
    
    imprimir_metricas(best_metricas)

    if best_model_weights is not None:
        modelo.load_state_dict(best_model_weights)
    modelo.eval()

    dummy_x_num  = x_num[[0]].to(dev)
    dummy_x_base = x_base[[0]].to(dev)
    dummy_x_add  = x_add[[0]].to(dev)

    torch.onnx.export(
        modelo, 
        (dummy_x_num, dummy_x_base, dummy_x_add), 
        onnx_path, 
        export_params=True,
        input_names=["x_num", "x_base", "x_add"], 
        output_names=["outputs"], 
        dynamic_axes={
            "x_num": {0: "batch_size"},
            "x_base": {0: "batch_size"},
            "x_add": {0: "batch_size"},
            "outputs": {0: "batch_size"}
        }
    )
    print(f"Modelo ONNX salvo em: {onnx_path}")


    return {
        "best_val_mse": best_metricas["mse"],
        "best_val_mae": best_metricas["mae"],
        "best_val_r2": best_metricas["r2"],
        "onnx_path": onnx_path}
    