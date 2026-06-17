import os 
from dotenv import load_dotenv
import pandas as pd 
import numpy as np

def carregar_dados() -> pd.DataFrame:
    """
    Carrega os dados do arquivo CSV especificado na variável de ambiente `CAMINHO_DOS_DADOS`.
    
    Retorna
    -------
    pandas.DataFrame
        DataFrame contendo os dados carregados.
    """
  
    load_dotenv()
    DATA_PATH = os.getenv("CAMINHO_DOS_DADOS")
    df = pd.read_csv(DATA_PATH, sep=';')
    return df

def preparar_dataframe(df):
    """
    Prepara os dados para processamento de soldagem.
    A função separa o DataFrame de acordo com o tipo de passe (raiz, enchimento e acabamento) e adiciona a coluna
    `tipo_de_peca`, utilizada para identificar se a peça corresponde a um tubo ou a uma chapa.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame contendo os dados de soldagem.

    Retorna
    -------
    pandas.DataFrame
        DataFrame processado, com os registros organizados por
        tipo de passe e com a informação de tipo de peça adicionada.
    """

    #Adiciona a coluna tipo_peca
    df['material1_diametro'] = pd.to_numeric(df['material1_diametro'], errors='coerce').fillna(0.0)
    df['tipo_peca'] = np.where(df['material1_diametro'] > 0, 'tubo', 'chapa')

    #Preparação dos dados
    colunas_comuns = [
        'normas_referencia', 'material1_pnumber', 'material1_especificacao',
        'material1_espessura', 'material1_diametro', 'tipo_peca',
        'tipo_chanfro', 'angulo', 'nariz', 'abertura_raiz', 'cobre_junta',
        'goivagem', 'posicao_peca', 'progressao', 'pre_aquecimento', 'temperatura_interpasse'
    ]
    passes = {
        'raiz': {
        'processo': 'raiz_processo', 'classificacao': 'raiz_classificacao', 'diametro': 'raiz_diametro',
        'polaridade': 'raiz_polaridade', 'limpeza': 'raiz_limpeza', 'voltagem': 'raiz_voltagem','amperagem': 'raiz_amperagem',
        'velocidade_de_soldagem': 'raiz_vel_soldagem', 'tipo_gas_tocha': 'tipo_gas_tocha_raiz','vazao_gas_tocha': 'vazao_gas_tocha_raiz',
        'pureza_gas_tocha': 'pureza_gas_tocha_raiz', 'tipo_gas_purga': 'tipo_gas_purga_raiz', 'vazao_gas_purga': 'vazao_gas_purga_raiz','pureza_gas_purga': 'pureza_gas_purga_raiz' },

        'enchimento': {
        'processo': 'enchimento_processo','classificacao': 'enchimento_classificacao','diametro': 'enchimento_diametro',
        'polaridade': 'enchimento_polaridade','limpeza': 'enchimento_limpeza','voltagem': 'enchimento_voltagem','amperagem': 'enchimento_amperagem',
        'velocidade_de_soldagem': 'enchimento_vel_soldagem','tipo_gas_tocha': 'tipo_gas_tocha_acabamento', 
        'vazao_gas_tocha': 'vazao_gas_tocha_acabamento','pureza_gas_tocha': 'pureza_gas_tocha_acabamento', 'tipo_gas_purga': 'tipo_gas_purga_acabamento',     
        'vazao_gas_purga': 'vazao_gas_purga_acabamento','pureza_gas_purga': 'pureza_gas_purga_acabamento'},

        'acabamento': {
        'processo': 'acabamento_processo','classificacao': 'acabamento_classificacao','diametro': 'acabamento_diametro',
        'polaridade': 'acabamento_polaridade','limpeza': 'acabamento_limpeza','voltagem': 'acabamento_voltagem',
        'amperagem': 'acabamento_amperagem','velocidade_de_soldagem': 'acabamento_vel_soldagem','tipo_gas_tocha': 'tipo_gas_tocha_acabamento',
        'vazao_gas_tocha': 'vazao_gas_tocha_acabamento','pureza_gas_tocha': 'pureza_gas_tocha_acabamento','tipo_gas_purga': 'tipo_gas_purga_acabamento',
        'vazao_gas_purga': 'vazao_gas_purga_acabamento','pureza_gas_purga': 'pureza_gas_purga_acabamento'}}

    # Unificação das tabelas por tipo de passe
    dfs = []
    for passe, cols in passes.items():
        temp = df[colunas_comuns].copy()
        temp['passe'] = passe
        temp['processo'] = df[cols['processo']]
        temp['classificacao'] = df[cols['classificacao']]
        temp['diametro_arame'] = df[cols['diametro']]
        temp['polaridade'] = df[cols['polaridade']]
        temp['limpeza'] = df[cols['limpeza']]
        temp['voltagem'] = df[cols['voltagem']]
        temp['amperagem'] = df[cols['amperagem']]
        temp['velocidade_de_soldagem'] = df[cols['velocidade_de_soldagem']]
        temp['tipo_gas_tocha'] = df[cols['tipo_gas_tocha']]
        temp['vazao_gas_tocha'] = df[cols['vazao_gas_tocha']]
        temp['pureza_gas_tocha'] = df[cols['pureza_gas_tocha']]   # Mapeado para o DataFrame unificado
        temp['tipo_gas_purga'] = df[cols['tipo_gas_purga']] 
        temp['vazao_gas_purga'] = df[cols['vazao_gas_purga']] 
        temp['pureza_gas_purga'] = df[cols['pureza_gas_purga']]   # Mapeado para o DataFrame unificado
        dfs.append(temp)
        
    dados = pd.concat(dfs, ignore_index=True)

    dados = dados.rename(columns={
        'classificacao': 'material_adicao',
        'material1_pnumber': 'pnumber',
        'material1_especificacao': 'material_base',
        'material1_espessura': 'espessura',
        'material1_diametro': 'diametro_base'})
    return dados 

def salvar_dados(df: pd.DataFrame) -> None:
    """
    Salva os dados processados no caminho definido pela variável de ambiente CAMINHO_DADOS_TRATADOS.
    """
    load_dotenv()
    OUTPUT_PATH = os.getenv("CAMINHO_DADOS_REESTRUTURADOS")
    print(f"Salvando em: {os.path.abspath(OUTPUT_PATH)}")
    df.to_csv( OUTPUT_PATH, sep=';', index=False )

def main() -> None: 
    print("Carregando dados...")
    dados = carregar_dados()
    print(f"Registros carregados: {len(dados)}")
    dados_reestruturados = preparar_dataframe(dados)
    print(f"Registros processados: {len(dados_reestruturados)}")
    salvar_dados(dados_reestruturados)
    print("Arquivo salvo com sucesso!")

if __name__ == "__main__":
    main()