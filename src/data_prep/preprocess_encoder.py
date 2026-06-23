import torch
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocessar_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara, divide e codifica os dados para o treinamento do modelo.

    O pipeline realiza as seguintes etapas:
    1. Separação das features (X) e alvos (y).
    2. Divisão em conjuntos de treino e teste (80/20) para evitar data leakage.
    3. Aplicação de StandardScaler em variáveis contínuas.
    4. Manutenção (passthrough) das variáveis de gases que já variam de 0 a 1.
    5. Aplicação de OneHotEncoder em variáveis categóricas de baixa cardinalidade.
    6. Criação de índices inteiros para materiais (base e adição) visando o uso  em camadas de Embedding de Redes Neurais, com tratamento para dados desconhecidos (UNK=0).

    Args:
        df (pd.DataFrame): DataFrame com os dados brutos de soldagem.

    Returns:
        tuple: Uma tupla contendo 10 elementos na seguinte ordem:
            - X_train_num_cat (np.ndarray): Matriz de treino com numéricas e one-hot.
            - X_test_num_cat (np.ndarray): Matriz de teste com numéricas e one-hot.
            - X_train_emb_base (np.ndarray): Array 1D de treino para o embedding do material base.
            - X_test_emb_base (np.ndarray): Array 1D de teste para o embedding do material base.
            - X_train_emb_add (np.ndarray): Array 1D de treino para o embedding do material de adição.
            - X_test_emb_add (np.ndarray): Array 1D de teste para o embedding do material de adição.
            - y_train (np.ndarray): Matriz com as 3 variáveis alvo de treino.
            - y_test (np.ndarray): Matriz com as 3 variáveis alvo de teste.
            - preprocessor (ColumnTransformer): O transformador ajustado (útil para dados futuros).
            - mappings (dict): Dicionário com os mapas dos materiais e tamanhos de vocabulário.
    """
    df = df.copy()
    
    target_cols = ['voltagem', 'amperagem', 'velocidade_de_soldagem']
    X = df.drop(columns=target_cols)
    y = df[target_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    
    features_numericas = ['espessura', 'diametro_base', 'angulo', 'nariz', 'abertura_raiz', 'pre_aquecimento', 'temperatura_interpasse', 'diametro_arame', 'vazao_gas_tocha', 'vazao_gas_purga']
    features_gases = ["tocha_AR", "tocha_CO2", "tocha_O2", "tocha_N2", "tocha_SemGas","purga_AR", "purga_CO2", "purga_O2", "purga_N2", "purga_SemGas"]
    features_onehot = ['tipo_peca', 'passe', 'goivagem', 'cobre_junta', 'polaridade', 'progressao', 'processo', 'normas_referencia', 'tipo_chanfro', 'posicao_peca', 'limpeza', 'pnumber']
    
    preprocessor = ColumnTransformer(   
        transformers=[
            ('num', StandardScaler(), features_numericas),
            ('gases', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),]), features_gases),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_onehot)
        ],
        remainder='drop') 

    X_train_num_cat = preprocessor.fit_transform(X_train)
    X_test_num_cat = preprocessor.transform(X_test)
    
    #Tratamento para Embeddings (Alta Cardinalidade)
    def criar_mapping(series):
        categorias = series.dropna().unique()
        return {cat: i+1 for i, cat in enumerate(categorias)}

    map_base = criar_mapping(X_train["material_base"])
    map_add = criar_mapping(X_train["material_adicao"])
    
    
    def aplicar_mapping(df_col, mapping):
        return df_col.map(lambda x: mapping.get(x, 0)).values

    X_train_emb_base = aplicar_mapping(X_train["material_base"], map_base)
    X_train_emb_add = aplicar_mapping(X_train["material_adicao"], map_add)
    X_test_emb_base = aplicar_mapping(X_test["material_base"], map_base)
    X_test_emb_add = aplicar_mapping(X_test["material_adicao"], map_add)

    # Agrupando os mapeamentos caso precise saber o tamanho do vocabulário para a Rede Neural
    mappings = {
        'material_base': map_base, 
        'material_adicao': map_add,
        'vocab_sizes': {
            'material_base': len(map_base) + 1, # +1 por causa do UNK
            'material_adicao': len(map_add) + 1
        }
    }

    return (
        X_train_num_cat, X_test_num_cat, 
        X_train_emb_base, X_test_emb_base, 
        X_train_emb_add, X_test_emb_add, 
        y_train_scaled, y_test_scaled, 
        preprocessor, mappings, y_scaler)

def salvar_dados_como_tensores(resultados_preprocessamento: tuple, nome_base_arquivo: str = 'dados_soldagem') -> None:
    """
    Converte as matrizes NumPy geradas pelo pré-processamento em tensores do PyTorch e salva os dados.

    Args:
        resultados_preproc (tuple): A tupla exata retornada pela função `preprocessar_encoder`, 
                                    contendo 10 elementos (matrizes de treino/teste, preprocessor e mappings).
        nome_base_arquivo (str, opcional): Nome base para os arquivos salvos. O padrão é 'dados_soldagem'.

    Returns:
        None: A função não retorna nada, apenas salva os arquivos no diretório atual:
              - {nome_base_arquivo}.pt (Contém o dicionário com todos os tensores)
              - {nome_base_arquivo}_preprocessor.joblib (Objeto do Scikit-Learn)
              - {nome_base_arquivo}_mappings.joblib (Dicionário de mapeamentos e vocabulário)
    """
    
    (X_train_num_cat, X_test_num_cat, X_train_emb_base, X_test_emb_base, X_train_emb_add, X_test_emb_add, y_train, y_test, preprocessor, mappings, y_scaler) = resultados_preprocessamento

    #Convertendo para tensores    
    X_train_num_tensor = torch.tensor(X_train_num_cat, dtype=torch.float32)
    X_test_num_tensor = torch.tensor(X_test_num_cat, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    X_train_emb_base_tensor = torch.tensor(X_train_emb_base, dtype=torch.long)
    X_test_emb_base_tensor = torch.tensor(X_test_emb_base, dtype=torch.long)
    X_train_emb_add_tensor = torch.tensor(X_train_emb_add, dtype=torch.long)
    X_test_emb_add_tensor = torch.tensor(X_test_emb_add, dtype=torch.long)

    
    caminho_tensores = f"{nome_base_arquivo}.pt"
    torch.save({
        'X_train_num': X_train_num_tensor, 'X_test_num': X_test_num_tensor,'X_train_emb_base': X_train_emb_base_tensor,
        'X_test_emb_base': X_test_emb_base_tensor,'X_train_emb_add': X_train_emb_add_tensor,'X_test_emb_add': X_test_emb_add_tensor,
        'y_train': y_train_tensor,'y_test': y_test_tensor}, caminho_tensores)
    
    # Salva os objetos utilitários usando joblib 
    caminho_prep = f"{nome_base_arquivo}_preprocessor.joblib"
    caminho_maps = f"{nome_base_arquivo}_mappings.joblib"
    caminho_yscaler = f"{nome_base_arquivo}_yscaler.joblib"
    joblib.dump(preprocessor, caminho_prep)
    joblib.dump(mappings, caminho_maps)
    joblib.dump(y_scaler, caminho_yscaler)
    
    print(f"\n✅ Concluído! Arquivos gerados com sucesso:")
    print(f"   -> {caminho_tensores}")
    print(f"   -> {caminho_prep}")
    print(f"   -> {caminho_maps}")
    print(f"   -> {caminho_yscaler}")

def main()-> None:
    caminho_dados = 'data/raw/dados_preprocessados.csv' 
    print(f"Iniciando o processamento do arquivo: {caminho_dados}")
    
    try:
        df_bruto = pd.read_csv(caminho_dados) 
        print("\nExecutando o pipeline de pré-processamento e codificação...")
        resultados = preprocessar_encoder(df_bruto)
        print("\nConvertendo para tensores PyTorch e salvando artefatos...")
        salvar_dados_como_tensores(resultados, nome_base_arquivo='dados_modelo_soldagem')
        
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo '{caminho_dados}' não foi encontrado no diretório atual.")
    except Exception as e:
        print(f"❌ Ocorreu um erro durante a execução: {e}")

if __name__ == "__main__":
    main()