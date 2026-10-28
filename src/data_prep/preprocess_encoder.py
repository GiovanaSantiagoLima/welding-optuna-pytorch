from pathlib import Path
import torch
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GroupShuffleSplit   # NOVO (no topo do arquivo; pode remover o import de train_test_split)

def preprocessar_encoder(df: pd.DataFrame) -> tuple:
    df = df.copy()

    target_cols = ['voltagem', 'amperagem', 'velocidade_de_soldagem']
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    # NOVO: split por grupo (mesma junta nos 3 passes fica sempre do mesmo lado)
    colunas_grupo = [
        'normas_referencia', 'pnumber', 'material_base', 'espessura', 'diametro_base',
        'tipo_peca', 'tipo_chanfro', 'angulo', 'nariz', 'abertura_raiz', 'cobre_junta',
        'goivagem', 'pre_aquecimento', 'temperatura_interpasse',
    ]  # sem posicao_peca/progressao, que mudam por passe
    grupo = X.groupby(colunas_grupo, dropna=False).ngroup().to_numpy()
    print(f"Grupos: {grupo.max() + 1} | linhas: {len(X)} | média de linhas/grupo: {len(X)/(grupo.max()+1):.2f}")

    idx_tr, idx_te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=grupo)
    )
    X_train, X_test = X.iloc[idx_tr], X.iloc[idx_te]
    y_train, y_test = y.iloc[idx_tr], y.iloc[idx_te]
    g_train = grupo[idx_tr]
    # FIM NOVO

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    features_numericas = ['espessura', 'diametro_base', 'angulo', 'nariz', 'abertura_raiz', 'pre_aquecimento', 'temperatura_interpasse', 'diametro_arame', 'vazao_gas_tocha', 'vazao_gas_purga']
    features_gases = ["tocha_AR", "tocha_CO2", "tocha_O2", "tocha_N2", "tocha_SemGas", "purga_AR", "purga_CO2", "purga_O2", "purga_N2", "purga_SemGas"]
    features_onehot = ['tipo_peca', 'passe', 'goivagem', 'cobre_junta', 'polaridade', 'progressao', 'processo', 'normas_referencia', 'tipo_chanfro', 'posicao_peca', 'limpeza', 'pnumber']

    gases_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_numericas),
            ('gases', gases_pipeline, features_gases),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_onehot)
        ],
        remainder='drop'
    )
    X_train_num_cat = preprocessor.fit_transform(X_train)
    X_test_num_cat = preprocessor.transform(X_test)

    # Tratamento para Embeddings (Alta Cardinalidade)
    NGRAM_SIZES = (1, 2)
    NUM_BUCKETS = 2048
    MAX_NGRAMS = 128

    from src.data_prep.material import encode
    def tokenizar(series):
        cache = {m: encode(str(m)) for m in series.dropna().unique()}
        vazio = encode("")
        return np.array([cache.get(m, vazio) for m in series], dtype=np.int64)

    X_train_emb_base = tokenizar(X_train["material_base"])
    X_train_emb_add  = tokenizar(X_train["material_adicao"])
    X_test_emb_base  = tokenizar(X_test["material_base"])
    X_test_emb_add   = tokenizar(X_test["material_adicao"])

    tokens = {"ngram": {"ngram_sizes": NGRAM_SIZES, "num_buckets": NUM_BUCKETS, "max_ngrams": MAX_NGRAMS}}

    return (
        X_train_num_cat, X_test_num_cat,
        X_train_emb_base, X_test_emb_base,
        X_train_emb_add, X_test_emb_add,
        y_train_scaled, y_test_scaled,
        preprocessor, tokens, y_scaler,
        g_train)   # NOVO: 12º item

def salvar_dados_como_tensores(resultados_preprocessamento: tuple, nome_base_arquivo: str = 'dados_soldagem') -> None:
    """
    Converte as matrizes NumPy geradas pelo pré-processamento em tensores do PyTorch e salva os dados.

    Args:
        resultados_preprocessamento (tuple): A tupla retornada por `preprocessar_encoder`,
            contendo 12 elementos (matrizes de treino/teste, preprocessor, tokens, y_scaler e g_train).
        nome_base_arquivo (str, opcional): Nome base dos arquivos salvos.

    Returns:
        None: Salva no disco:
              - {nome_base_arquivo}.pt (dicionário com todos os tensores)
              - preprocessor_v1.joblib, token_v1.joblib, yscaler_v1.joblib
    """
    BASE_DIR = Path(__file__).resolve().parents[2]
    (X_train_num_cat, X_test_num_cat,
     X_train_emb_base, X_test_emb_base,
     X_train_emb_add, X_test_emb_add,
     y_train, y_test,
     preprocessor, tokens, y_scaler, g_train) = resultados_preprocessamento

    # Convertendo para tensores
    X_train_num_tensor = torch.tensor(X_train_num_cat, dtype=torch.float32)
    X_test_num_tensor = torch.tensor(X_test_num_cat, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    X_train_emb_base_tensor = torch.tensor(X_train_emb_base, dtype=torch.long)
    X_test_emb_base_tensor = torch.tensor(X_test_emb_base, dtype=torch.long)
    X_train_emb_add_tensor = torch.tensor(X_train_emb_add, dtype=torch.long)
    X_test_emb_add_tensor = torch.tensor(X_test_emb_add, dtype=torch.long)
    g_train_tensor = torch.tensor(g_train, dtype=torch.long)   # NOVO: grupos para o GroupKFold

    caminho_saida_tensores = BASE_DIR / "data" / "processed"
    caminho_saida_artefatos = BASE_DIR / "data" / "artifacts"
    caminho_saida_tensores.mkdir(parents=True, exist_ok=True)
    caminho_saida_artefatos.mkdir(parents=True, exist_ok=True)

    caminho_tensores = caminho_saida_tensores / f"{nome_base_arquivo}.pt"
    caminho_prep = caminho_saida_artefatos / "preprocessor_v1.joblib"
    caminho_token = caminho_saida_artefatos / "token_v1.joblib"
    caminho_yscaler = caminho_saida_artefatos / "yscaler_v1.joblib"

    torch.save({
        'X_train_num': X_train_num_tensor, 'X_test_num': X_test_num_tensor,
        'X_train_emb_base': X_train_emb_base_tensor, 'X_test_emb_base': X_test_emb_base_tensor,
        'X_train_emb_add': X_train_emb_add_tensor, 'X_test_emb_add': X_test_emb_add_tensor,
        'y_train': y_train_tensor, 'y_test': y_test_tensor,
        'g_train': g_train_tensor,                              # NOVO
    }, caminho_tensores)

    joblib.dump(preprocessor, caminho_prep)
    joblib.dump(tokens, caminho_token)
    joblib.dump(y_scaler, caminho_yscaler)

    print(f"\n✅ Concluído! Arquivos gerados com sucesso:")
    print(f"   -> {caminho_tensores}")
    print(f"   -> {caminho_prep}")
    print(f"   -> {caminho_token}")
    print(f"   -> {caminho_yscaler}")

def main()-> None:
    DIRETORIO_RAIZ = Path(__file__).resolve().parents[2]
    CAMINHO_FICHEIRO = DIRETORIO_RAIZ / "data" / "processed" / "dados_preprocessados.csv"
    print(f"Iniciando o processamento do arquivo: {CAMINHO_FICHEIRO}")
    
    try:
        df_bruto = pd.read_csv(CAMINHO_FICHEIRO) 
        print("\nExecutando o pipeline de pré-processamento e codificação...")
        resultados = preprocessar_encoder(df_bruto)
        print("\nConvertendo para tensores PyTorch e salvando artefatos...")
        salvar_dados_como_tensores(resultados, nome_base_arquivo='dados_modelo_soldagem')
        
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo '{CAMINHO_FICHEIRO}' não foi encontrado no diretório atual.")
    except Exception as e:
        print(f"❌ Ocorreu um erro durante a execução: {e}")


if __name__ == "__main__":

    main()