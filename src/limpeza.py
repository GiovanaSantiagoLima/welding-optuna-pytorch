import re
import numpy as np
import pandas as pd


#Funções auxiliares para limpeza e tratamento dos dados
def limpar_vazao(val):
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

def limpar_normas_referencia(texto):

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

def ajustar_progressao(linha):
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

def ajustar_posicao(linha):
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

def limpeza(df_bruto: pd.DataFrame) -> pd.DataFrame:

    df = df_bruto.copy()

    # 1. PREPARAÇÃO E UNIFICAÇÃO DOS DADOS (Por tipo de passe)
    df['material1_diametro'] = pd.to_numeric(df['material1_diametro'], errors='coerce').fillna(0.0)
    df['tipo_peca'] = np.where(df['material1_diametro'] > 0, 'tubo', 'chapa')
    colunas_comuns = [
        'normas_referencia', 'material1_pnumber', 'material1_especificacao',
        'material1_espessura', 'material1_diametro', 'tipo_peca',
        'tipo_chanfro', 'angulo', 'nariz', 'abertura_raiz', 'cobre_junta',
        'goivagem', 'posicao_peca', 'progressao', 'pre_aquecimento', 'temperatura_interpasse'
    ]
    passes = {
        'raiz': {
            'processo': 'raiz_processo', 'classificacao': 'raiz_classificacao', 'diametro': 'raiz_diametro',
            'polaridade': 'raiz_polaridade', 'limpeza': 'raiz_limpeza', 'voltagem': 'raiz_voltagem',
            'amperagem': 'raiz_amperagem', 'velocidade_de_soldagem': 'raiz_vel_soldagem',
            'tipo_gas_tocha': 'tipo_gas_tocha_raiz', 'vazao_gas_tocha': 'vazao_gas_tocha_raiz',
            'tipo_gas_purga': 'tipo_gas_purga_raiz', 'vazao_gas_purga': 'vazao_gas_purga_raiz'
        },
        'enchimento': {
            'processo': 'enchimento_processo', 'classificacao': 'enchimento_classificacao', 'diametro': 'enchimento_diametro',
            'polaridade': 'enchimento_polaridade', 'limpeza': 'enchimento_limpeza', 'voltagem': 'enchimento_voltagem',
            'amperagem': 'enchimento_amperagem', 'velocidade_de_soldagem': 'enchimento_vel_soldagem',
            'tipo_gas_tocha': 'tipo_gas_tocha_acabamento', 'vazao_gas_tocha': 'vazao_gas_tocha_acabamento',
            'tipo_gas_purga': 'tipo_gas_purga_acabamento', 'vazao_gas_purga': 'vazao_gas_purga_acabamento'
        },
        'acabamento': {
            'processo': 'acabamento_processo', 'classificacao': 'acabamento_classificacao', 'diametro': 'acabamento_diametro',
            'polaridade': 'acabamento_polaridade', 'limpeza': 'acabamento_limpeza', 'voltagem': 'acabamento_voltagem',
            'amperagem': 'acabamento_amperagem', 'velocidade_de_soldagem': 'acabamento_vel_soldagem',
            'tipo_gas_tocha': 'tipo_gas_tocha_acabamento', 'vazao_gas_tocha': 'vazao_gas_tocha_acabamento',
            'tipo_gas_purga': 'tipo_gas_purga_acabamento', 'vazao_gas_purga': 'vazao_gas_purga_acabamento'
        }
    }
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
        temp['tipo_gas_purga'] = df[cols['tipo_gas_purga']]
        temp['vazao_gas_purga'] = df[cols['vazao_gas_purga']]
        dfs.append(temp)
        
    dados = pd.concat(dfs, ignore_index=True)
    dados = dados.rename(columns={
        'classificacao': 'material_adicao',
        'material1_pnumber': 'pnumber',
        'material1_especificacao': 'material_base',
        'material1_espessura': 'espessura',
        'material1_diametro': 'diametro_base'
    })

    # 2. TRATAMENTO DE DADOS AUSENTES
    dados['goivagem'] = dados['goivagem'].fillna('Informação Desconhecida')
    dados['cobre_junta'] = dados['cobre_junta'].fillna('Informação Desconhecida')
    dados['vazao_gas_purga'] = dados['vazao_gas_purga'].apply(limpar_vazao)
    dados['vazao_gas_tocha'] = dados['vazao_gas_tocha'].apply(limpar_vazao)
    dados['pre_aquecimento'] = dados['pre_aquecimento'].fillna(25.0)
    dados['temperatura_interpasse'] = dados['temperatura_interpasse'].fillna(25.0)
    for col in ['nariz', 'abertura_raiz', 'angulo']:
        dados[col] = dados[col].fillna(0.0)
    dados['diametro_arame'] = dados['diametro_arame'].fillna(0.0)
    dados['pnumber'] = dados['pnumber'].fillna('Não Informado').astype(str)
    dados['posicao_peca'] = dados['posicao_peca'].fillna('Não Informado').astype(str)
    dados['espessura'] = dados['espessura'].fillna(dados['espessura'].median())
    dados = dados.dropna(subset=['voltagem', 'amperagem', 'velocidade_de_soldagem'])
    
   
    # 3. TRATAMENTO TEXTUAL DOS DADOS
    dados['normas_referencia'] = dados['normas_referencia'].apply(limpar_normas_referencia)
    mapeamento_limpeza = {
        'ESMER.': 'ESMERILHAMENTO', 'ESMER': 'ESMERILHAMENTO', 'ESMERILHAMENTO': 'ESMERILHAMENTO',
        'ESMERIL.': 'ESMERILHAMENTO', 'LIXAMENTO': 'LIXAMENTO', 'ESCOVAMENTO': 'ESCOVAMENTO',
        'ESCOV.': 'ESCOVAMENTO', 'ESCOV': 'ESCOVAMENTO', 'ESMER./ESCOV': 'ESMERILHAMENTO + ESCOVAMENTO',
        'ESCOVAMENTO/ESMERILHAMENTO': 'ESMERILHAMENTO + ESCOVAMENTO', 'DISCO ABRASIVO/ESCOVAMENTO': 'ESMERILHAMENTO + ESCOVAMENTO',
        'DISCO/ESCOVAMENTO': 'ESMERILHAMENTO + ESCOVAMENTO', 'ESMERILHAMENTO/LIXAMENTO': 'ESMERILHAMENTO + LIXAMENTO',
        'ESCOV./ESMER.': 'ESMERILHAMENTO + ESCOVAMENTO', 'ESCOV./ESMERIL.': 'ESMERILHAMENTO + ESCOVAMENTO',
        'ESCOV/ESMER': 'ESMERILHAMENTO + ESCOVAMENTO', 'ESMER/ESCOV': 'ESMERILHAMENTO + ESCOVAMENTO',
        'ESMERILHAMENTO/ESCOVAMENTO': 'ESMERILHAMENTO + ESCOVAMENTO', 'DISCO ABRASIVO + ESCOVA AÇO': 'ESMERILHAMENTO + ESCOVAMENTO',
        'DISCO E ESCOVA DE INOX': 'ESMERILHAMENTO + ESCOVAMENTO', 'ESMER./ESCOV.': 'ESMERILHAMENTO + ESCOVAMENTO',
        'ESCOV./SOLVENTE': 'ESCOVAMENTO + SOLVENTE', 'INICIAL: AO METAL BRILHANTE': 'METAL BRILHANTE',
        'AO METAL BRILHANTE': 'METAL BRILHANTE', 'NAN': np.nan, 'nan': np.nan, 'OBS1': np.nan, 'N/A': np.nan
    }
    dados['limpeza'] = dados['limpeza'].astype(str).str.strip().map(mapeamento_limpeza)
    dados['limpeza'] = dados['limpeza'].fillna('Informação Desconhecida')
    dados = dados[~dados['processo'].isin(['SAW'])]
    dados = dados[~dados['processo'].str.contains('TIG/ER', case=False, na=False)]
    dados['progressao'] = dados['progressao'].astype(str).str.strip().str.upper()
    dados['progressao'] = dados['progressao'].replace({'NAN': np.nan, 'NAN ': np.nan, 'nan': np.nan})
    dados['progressao'] = dados.apply(ajustar_progressao, axis=1)
    dados['progressao'] = dados['progressao'].fillna('Informação Desconhecida')
    dados['posicao_peca'] = dados['posicao_peca'].astype(str).str.strip().str.replace(" e ", "/", case=False, regex=False).str.upper()
    dados['posicao_peca'] = dados['posicao_peca'].replace({"NAN": np.nan, "NÃO INFORMADO": np.nan})
    dados['posicao_peca'] = dados.apply(ajustar_posicao, axis=1)
    dados['posicao_peca'] = dados['posicao_peca'].fillna("Informação Desconhecida")
    dados['polaridade'] = dados['polaridade'].replace('NAN', 'Informação Desconhecida')
    for gas_col in ['tipo_gas_tocha', 'tipo_gas_purga']:
        dados[gas_col] = dados[gas_col].replace('NAN', 'Sem Gás')
        
    # 4. TRATAMENTO E CONVERSÃO DOS ALVOS (TARGETS)
    dados['amperagem'] = pd.to_numeric(dados['amperagem'].astype(str).str.replace(',', '.'), errors='coerce')
    dados['velocidade_de_soldagem'] = pd.to_numeric(dados['velocidade_de_soldagem'], errors='coerce')
    dados['voltagem'] = pd.to_numeric(dados['voltagem'], errors='coerce')
    
    if 478 in dados.index:
        dados.loc[478, 'amperagem'] = 179.0    

    return dados