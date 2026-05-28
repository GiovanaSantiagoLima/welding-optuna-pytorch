import os
from pathlib import Path
import numpy as np
import category_encoders as ce
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def carregar_e_processar_dados():
    csv_path = Path(__file__).resolve().parents[2] / "data" / "all_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em: {csv_path}")
        
    df = pd.read_csv(csv_path, sep=';')
    
    # 1. Seleção e empilhamento dos passes de soldagem
    colunas_comuns = [
        'material1_pnumber', 'material1_especificacao', 'normas_referencia', 
        'tipo_chanfro', 'angulo', 'nariz', 'abertura_raiz', 'cobre_junta', 
        'goivagem'
    ]
    
    dados_raiz = df[colunas_comuns + [
        'raiz_processo', 'raiz_classificacao', 'raiz_diametro', 
        'raiz_voltagem', 'raiz_amperagem', 'raiz_vel_soldagem'
    ]].copy()
    dados_raiz['passe'] = 'raiz'
    dados_raiz.rename(columns={
        'raiz_processo': 'processo', 'raiz_classificacao': 'consumivel', 'raiz_diametro': 'diametro_consumivel',
        'raiz_voltagem': 'voltagem', 'raiz_amperagem': 'amperagem', 'raiz_vel_soldagem': 'velocidade_soldagem'
    }, inplace=True)
    
    dados_enchimento = df[colunas_comuns + [
        'enchimento_processo', 'enchimento_classificacao', 'enchimento_diametro', 
        'enchimento_voltagem', 'enchimento_amperagem', 'enchimento_vel_soldagem'
    ]].copy()
    dados_enchimento['passe'] = 'enchimento'
    dados_enchimento.rename(columns={
        'enchimento_processo': 'processo', 'enchimento_classificacao': 'consumivel', 'enchimento_diametro': 'diametro_consumivel',
        'enchimento_voltagem': 'voltagem', 'enchimento_amperagem': 'amperagem', 'enchimento_vel_soldagem': 'velocidade_soldagem'
    }, inplace=True)
    
    dados_acabamento = df[colunas_comuns + [
        'acabamento_processo', 'acabamento_classificacao', 'acabamento_diametro', 
        'acabamento_voltagem', 'acabamento_amperagem', 'acabamento_vel_soldagem'
    ]].copy()
    dados_acabamento['passe'] = 'acabamento'
    dados_acabamento.rename(columns={
        'acabamento_processo': 'processo', 'acabamento_classificacao': 'consumivel', 'acabamento_diametro': 'diametro_consumivel',
        'acabamento_voltagem': 'voltagem', 'acabamento_amperagem': 'amperagem', 'acabamento_vel_soldagem': 'velocidade_soldagem'
    }, inplace=True)
    
    df_final = pd.concat([dados_raiz, dados_enchimento, dados_acabamento], ignore_index=True)
    
    # 2. Padronização de Strings e Tratamento de Nulos
    mapeamento_processos = {
        'TIG ': 'TIG', 'TIG': 'TIG', 'SMAW': 'ER', 'ER': 'ER', 'ER309L': 'ER',
        'GMAW': 'MIG/MAG', 'MIG': 'MIG/MAG', 'TIG/ER': 'TIG/ER', 'FCAW': 'FCAW', 'SAW': 'SAW'
    }
    df_final['processo'] = df_final['processo'].replace(mapeamento_processos)
    df_final['goivagem'] = df_final['goivagem'].fillna('Sem goivagem')
    df_final['cobre_junta'] = df_final['cobre_junta'].fillna('Sem Cobre Junta')
    
    # Preenchimento de Nulos Numéricos baseado na mediana
    colunas_num = ['angulo', 'nariz', 'abertura_raiz', 'diametro_consumivel']
    for col in colunas_num:
        df_final[col] = df_final.groupby(['processo', 'passe'])[col].transform(lambda x: x.fillna(x.median()))
    df_final.dropna(subset=['voltagem', 'amperagem', 'velocidade_soldagem'], inplace=True)
    df_final.dropna(subset=colunas_num, inplace=True)
    

    print("🔀 Aplicando Encodings nas colunas categóricas...")
    
    colunas_categoricas = ['consumivel', 'material1_especificacao', 'material1_pnumber', 'tipo_chanfro']
    
    # Aplica o BinaryEncoder para colunas com muito texto (como o pnumber que tinha '5B')
    encoder = ce.BinaryEncoder(cols=colunas_categoricas)
    df_final = encoder.fit_transform(df_final)

    # Mapeamento manual do passe de solda
    mapeamento_passe = {'raiz': 1, 'enchimento': 2, 'acabamento': 3}
    df_final['passe'] = df_final['passe'].map(mapeamento_passe)
    
    # One-Hot Encoding para colunas com poucas categorias (Sim/Não/Normas)
    df_final = pd.get_dummies(df_final, columns=['goivagem', 'cobre_junta', 'normas_referencia', 'processo'])
    # -----------------------------------------------------------------

    # 2. SEPARAÇÃO DE ENTRADAS (X) E SAÍDAS (y)
    print("✂️ Separando features de entrada (X) e alvos (y)...")
    
    # Lista com as colunas que são o seu objetivo prever (as saídas)
    colunas_saida = ['voltagem', 'amperagem', 'velocidade_soldagem']
    
    # y_df pega apenas as colunas de saída
    y_df = df_final[colunas_saida].copy()
    
    # X_df pega todo o resto, excluindo as colunas de saída e IDs inúteis se houver
    X_df = df_final.drop(columns=colunas_saida)
    
    return X_df, y_df
