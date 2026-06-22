#Bibliotecas Utilizadas
import re
import numpy as np
import pandas as pd

#Funções Auxiliares
def limpar_vazao(val) -> float:
    """
    Converte valores de vazão para formato numérico.
    Trata valores ausentes, intervalos e textos contendo números.
    Quando múltiplos valores são encontrados, retorna a média.
    Args:
        val: Valor original da vazão.
    Returns:
        float: Valor de vazão convertido.
    """
    if pd.isna(val):
        return 0.0
    val_str = str(val).lower().strip()
    val_str = val_str.replace('dez', '12')
    val_str = val_str.replace(' a ', '/').replace('-', '/')
    numeros = re.findall(r'\d+\.\d+|\d+', val_str)
    if not numeros:
        return 0.0
    valores_float = [float(n) for n in numeros]
    return sum(valores_float) / len(valores_float)

def limpar_normas_referencia(texto)-> str:
    """
    Padroniza os valores da coluna de normas de referência.
    Args:
        texto: Texto original contendo a norma.
    Returns:
        str: Norma padronizada.
    """
    if pd.isna(texto):
        return 'Não Informado'
    texto = str(texto).lower()
    if re.search(r'\baws\b', texto):
        return 'AWS'
    if re.search(r'\basme\b.*\bix\b', texto) or re.search(r'\bix\b.*\basme\b', texto):
        return 'ASME IX'
    if re.search(r'\babs\b', texto):
        return 'ABS'
    if re.search(r'\bastm\b', texto):
        return 'ASME IX'
    return 'Outros'

def limpar_limpeza(texto) -> str:
    """
    Padroniza os valores da coluna de limpeza.
    Args:
        texto: Valor original da limpeza.
    Returns:
        str: Categoria padronizada.
    """
    if pd.isna(texto):
        return "Informação Desconhecida"
    texto = str(texto).upper().strip()
    if re.search(r"ESMER", texto):
        return "ESMERILHAMENTO"
    if re.search(r"LIXAMENTO", texto):
        return "LIXAMENTO"
    if re.search(r"ESCOV", texto):
        return "ESCOVAMENTO"
    if re.search(r"METAL BRILHANTE", texto):
        return "METAL BRILHANTE"
    if re.search(r"SOLVENTE", texto):
        return "ESCOVAMENTO + SOLVENTE"
    if re.search(r"ESMER.*ESCOV|ESCOV.*ESMER", texto):
        return "ESMERILHAMENTO + ESCOVAMENTO"
    if re.search(r"ESMER.*LIX|LIX.*ESMER", texto):
        return "ESMERILHAMENTO + LIXAMENTO"

    return "Outros"


def ajustar_progressao(linha) -> str:
    """
    Ajusta o valor de progressão conforme o passe de soldagem.
    Args:
    linha: Linha do DataFrame contendo as colunas 'progressao' e 'passe'.
    Returns:
    str: Valor de progressão ajustado.
    """
    prog = linha['progressao']
    passe = linha['passe']
    if pd.isna(prog): return np.nan    
    if '/' in prog:
        partes = prog.split('/') 
        if passe == 'raiz':
            return 'DESCENDENTE' if 'DESCENDENTE' in partes[0] else 'ASCENDENTE'
        else:
            return 'ASCENDENTE' if 'ASCENDENTE' in partes[1] else 'DESCENDENTE'       
    if '+' in prog and passe == 'acabamento':
        return 'DESCENDENTE' 
    return prog

def ajustar_posicao(linha)-> str:
    """
    Ajusta o valor da posição da peça conforme o passe de soldagem.
    Args:
    linha: Linha do DataFrame contendo as colunas 'posicao' e 'passe'.
    Returns:
    str: Valor de progressão ajustado.
    """
    posicao = linha['posicao_peca']
    passe = linha['passe']
    if pd.isna(posicao) or posicao == "NAN": return np.nan
    if "/" in posicao:
        partes = posicao.split("/")
        if passe == "raiz":
            return partes[0].strip()
        elif passe in ["enchimento", "acabamento"]:
            return partes[1].strip()
    return posicao

def preencher_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores ausentes em colunas específicas do DataFrame.
    Args:
        df: DataFrame contendo os dados de soldagem.
    Returns:
        pd.DataFrame: DataFrame com os valores ausentes tratados.
    """
    df = df.copy()
    df['goivagem'] = df['goivagem'].fillna('Informação Desconhecida')
    df['cobre_junta'] = df['cobre_junta'].fillna('Informação Desconhecida')
    df['pureza_gas_tocha'] = df['pureza_gas_tocha'].fillna('Sem Gás')
    df['pureza_gas_purga'] = df['pureza_gas_purga'].fillna('Sem Gás')
    df["polaridade"] = df["polaridade"].replace("NAN", "Informação Desconhecida")
    df['pre_aquecimento'] = df['pre_aquecimento'].fillna(25.0)
    for gas_col in ['tipo_gas_tocha', 'tipo_gas_purga']:
        df[gas_col] = df[gas_col].replace('NAN', 'Sem Gás')
    return df

def tratar_vazao(df: pd.DataFrame) -> pd.DataFrame:
   """Converte as colunas de vazão de gás para formato numérico aplicando a função limpar_vazão nas colunas de vazão de tocha e purga
   Args:
        df: DataFrame contendo os dados de soldagem.
    Returns:
        pd.DataFrame: DataFrame com as vazões tratadas.
   """
   df = df.copy()
   colunas = ["vazao_gas_tocha", "vazao_gas_purga"]
   for col in colunas:
        df[col] = df[col].apply(limpar_vazao)
   return df
    
def limpar_texto(df:pd.DataFrame)-> pd.DataFrame:
    """
    Realiza limpeza e padronização de colunas textuais do DataFrame.
    Args:
        df: DataFrame bruto.
    Returns:
        pd.DataFrame: DataFrame com textos padronizados.
    """
    df = df.copy()
    df['normas_referencia'] = df['normas_referencia'].apply(limpar_normas_referencia)
    df['limpeza'] = df['limpeza'].apply(limpar_limpeza)
    df["progressao"] = (
        df["progressao"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "NAN ": np.nan, "nan": np.nan})
    )

    df["progressao"] = df.apply(ajustar_progressao, axis=1)
    df["progressao"] = df["progressao"].fillna("Informação Desconhecida")

    df["posicao_peca"] = df.apply(ajustar_posicao, axis=1)
    df["posicao_peca"] = df["posicao_peca"].fillna("Informação Desconhecida")

    return df

def excluir_linhas_indesejadas(df:pd.DataFrame)-> pd.DataFrame:
    """
    Remove linhas indesejadas e registros com valores críticos ausentes.

    Critérios de exclusão:
        - Remove processos do tipo 'SAW'
        - Remove processos que contenham 'TIG/ER'
        - Remove linhas com valores ausentes em colunas críticas

    Args:
        df: DataFrame original.
    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    df = df.copy()
    # remover processos indesejados
    df = df[~df["processo"].isin(["SAW"])]
    df = df[~df["processo"].str.contains("TIG/ER", case=False, na=False)]
    # colunas críticas (não podem ter NaN)
    colunas_criticas = ["voltagem","amperagem", "velocidade_de_soldagem","nariz", "abertura_raiz","angulo", "diametro_arame","espessura"]
    df = df.dropna(subset=colunas_criticas)
    return df

#Pré Processamento dos gases    
def preprocessar_dados_gases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai composição dos gases usando valores reais de pureza.
    Cria variáveis numéricas de pureza dos gases e indicadores de presença.

    Args:
        df: DataFrame com dados brutos.
    Returns:
        pd.DataFrame: DataFrame com features de gases.
    """
    df = df.copy()

    gases = ["AR", "CO2", "O2", "N2"]

    def processar_gas(row, pref):
        tipo = str(row[f"tipo_gas_{pref}"])
        pureza = str(row[f"pureza_gas_{pref}"])
        res = {}
        gases_encontrados = re.findall(r"AR|CO2|O2|N2", tipo)
        valores = [float(n) for n in re.findall(r"\d+\.?\d*", pureza)]
        for g in gases:
            res[f"{pref}_{g}"] = np.nan
            res[f"{pref}_{g}_presente"] = 0
        for g, v in zip(gases_encontrados, valores):
            res[f"{pref}_{g}"] = v
            res[f"{pref}_{g}_presente"] = 1
        res[f"{pref}_SemGas"] = int("Sem Gás" in tipo or tipo in ["0", "nan"])
        return pd.Series(res)
    
    for p in ["tocha", "purga"]:
        df = df.join(df.apply(processar_gas, axis=1, args=(p,)))

    df = df.drop(columns=["tipo_gas_tocha","pureza_gas_tocha","tipo_gas_purga","pureza_gas_purga"],errors="ignore")
    return df

def main():
    nome_arquivo = 'data/dados_reestruturados.csv'
    print(f"Carregando os dados do arquivo: {nome_arquivo}...")
    try:
        df = pd.read_csv(nome_arquivo, sep=';') 
    except FileNotFoundError:
        print(f"Erro: O arquivo '{nome_arquivo}' não foi encontrado no diretório.")
        return

    print("Iniciando a limpeza e pré-processamento...")
    df_processado = limpar_texto(df)
    df_processado = preencher_nulos(df_processado)
    df_processado = tratar_vazao(df_processado)
    df_processado = excluir_linhas_indesejadas(df_processado)
    df_processado = preprocessar_dados_gases(df_processado)

    arquivo_saida = 'data/dados_preprocessados.csv'
    df_processado.to_csv(arquivo_saida, index=False)

    print(f"Dados salvos como '{arquivo_saida}'.")
    print(f"Total de linhas após a limpeza: {len(df_processado)}")


if __name__ == "__main__":
    main()
