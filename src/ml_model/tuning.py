import os
import json
import optuna
from dotenv import load_dotenv
from train import treinar, imprimir_metricas, OUTPUT_NAMES

load_dotenv()
DATA_PATH = os.getenv("DADOS_PROJETO")

# ── Configurações do estudo ───────────────────────────────────────────────────
ntrials_   = 50
epochs_    = 100
patience_  = 15
study_name = "soldagem_v1"


def objective(trial: optuna.Trial) -> float:
    """
    Função objetivo: retorna o melhor val_mse global do trial.
    O Optuna minimiza esse valor.
    """
    params = {
        "hidden_size":   trial.suggest_categorical("hidden_size", [128, 256]),
        "num_layers":    trial.suggest_int("num_layers", 2, 3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "dropout_rate":  trial.suggest_float("dropout_rate", 0.0, 0.15),
        "emb_dim":       trial.suggest_categorical("emb_dim", [4, 8]),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128]),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True),
        "activation":    trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
        "optimizer":     trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD", "RMSprop"]),
        "criterion":     trial.suggest_categorical("criterion", ["MSELoss", "L1Loss", "HuberLoss"]),
    }

    resultado = treinar(
        params=params,
        data_path=DATA_PATH,
        epochs=epochs_,
        patience=patience_,
    )

    return resultado["best_val_mse"]


def rodar_estudo():
    """Cria ou continua o estudo Optuna e salva os resultados."""

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=ntrials_, show_progress_bar=True, gc_after_trial=True)

    melhor = study.best_trial
    sep = "=" * 60

    # — Cabeçalho —
    print(f"\n{sep}")
    print(f"  MELHOR TRIAL: #{melhor.number}")
    print(sep)

    # — Parâmetros —
    print("  Parâmetros:")
    for k, v in melhor.params.items():
        print(f"    {k}: {v}")

    # — Métricas detalhadas do melhor trial —
    # Roda uma vez mais o melhor trial para obter MSE/MAE/R² por output
    print(f"\n  Calculando métricas detalhadas do melhor trial...")
    resultado_melhor = treinar(
        params=melhor.params,
        data_path=DATA_PATH,
        epochs=epochs_,
        patience=patience_,
    )

    print(f"\n  MÉTRICAS DO MELHOR TRIAL (val)")
    metricas_display = {k.replace("best_val_", ""): v for k, v in resultado_melhor.items() if k.startswith("best_val_")}
    imprimir_metricas(metricas_display, prefixo="  ")

    # — Importância dos hiperparâmetros —
    try:
        importancias = optuna.importance.get_param_importances(study)
        print(f"  Importância dos hiperparâmetros:")
        for param, imp in importancias.items():
            barra = "█" * int(imp * 40)
            print(f"    {param:<20} {barra} {imp:.3f}")
    except Exception:
        pass

    print(sep)

    # — Salva JSON —
    resultado_json = {
        "trial_number":             melhor.number,
        "best_val_mse":             resultado_melhor["best_val_mse"],
        "best_val_mae":             resultado_melhor["best_val_mae"],
        "best_val_r2":              resultado_melhor["best_val_r2"],
        "best_val_mse_voltagem":    resultado_melhor["best_val_mse_voltagem"],
        "best_val_mae_voltagem":    resultado_melhor["best_val_mae_voltagem"],
        "best_val_r2_voltagem":     resultado_melhor["best_val_r2_voltagem"],
        "best_val_mse_amperagem":   resultado_melhor["best_val_mse_amperagem"],
        "best_val_mae_amperagem":   resultado_melhor["best_val_mae_amperagem"],
        "best_val_r2_amperagem":    resultado_melhor["best_val_r2_amperagem"],
        "best_val_mse_velocidade":  resultado_melhor["best_val_mse_velocidade"],
        "best_val_mae_velocidade":  resultado_melhor["best_val_mae_velocidade"],
        "best_val_r2_velocidade":   resultado_melhor["best_val_r2_velocidade"],
        "params":                   melhor.params,
    }
    with open("melhor_trial.json", "w") as f:
        json.dump(resultado_json, f, indent=2)
    print("\n  Melhores parâmetros e métricas salvos em melhor_trial.json")

    return study


def treinar_modelo_final(json_path: str = "melhor_trial.json"):
    """
    Treina o modelo final com os melhores hiperparâmetros encontrados,
    usando mais épocas e patience maior.
    """
    with open(json_path) as f:
        dados = json.load(f)

    params = dados["params"]
    sep = "=" * 60

    print(f"{sep}")
    print("  Treinando modelo final com os melhores hiperparâmetros:")
    for k, v in params.items():
        print(f"    {k}: {v}")
    print(sep)

    resultado = treinar(
        params=params,
        data_path=DATA_PATH,
        epochs=300,
        patience=30,
    )

    print(f"\n  MÉTRICAS FINAIS DO MODELO (val)")
    metricas_display = {k.replace("best_val_", ""): v for k, v in resultado.items() if k.startswith("best_val_")}
    imprimir_metricas(metricas_display, prefixo="  ")
    print(f"  Parou na época: {resultado['stopped_epoch']}")
    print(sep)

    return resultado


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modo",
        choices=["tuning", "final"],
        default="tuning",
        help="'tuning' roda o Optuna | 'final' treina com melhor_trial.json",
    )
    args = parser.parse_args()

    if args.modo == "tuning":
        rodar_estudo()
    else:
        treinar_modelo_final()