# Previsão de Parâmetros de Soldagem com PyTorch e Optuna

Este repositório contém uma solução de Machine Learning / Deep Learning para **regressão multi-output** aplicada à previsão de parâmetros de soldagem (Voltagem, Amperagem e Velocidade de Soldagem).

A arquitetura utiliza **PyTorch** com camadas de *Embedding* para variáveis categóricas de alta cardinalidade, validação cruzada (*K-Fold*), otimização de hiperparâmetros automatizada via **Optuna** e exportação do modelo final para o formato **ONNX**.

---

## 🛠️ Estrutura do Repositório

```text
welding-optuna-pytorch/
├── data/
│   ├── artifacts/            # Scalers, encoders e mappings (.joblib)
│   ├── processed/            # Datasets tratados e tensores PyTorch (.pt)
│   └── raw/                  # Datasets brutos originais (.csv)
├── model/                    # Modelo final (.onnx) e parâmetros do melhor trial (.json)
├── notebook/                 # Análises exploratórias e estudos de hiperparâmetros (.ipynb)
├── src/
│   ├── data_prep/            # Pipeline de carregamento, limpeza e codificação de dados
│   └── ml_model/             # Dataset, arquitetura da rede, treino e tuning
├── .env                      # Variáveis de ambiente
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📌 Funcionalidades Principais

* **Pipeline de Dados (`src/data_prep/`)**:

  * **Carregamento e Estruturação**: Organização e unificação dos dados por passes (raiz, enchimento e acabamento) e identificação de geometrias (tubo vs. chapa).
  * **Limpeza e Imputação**: Padronização de normas de referência, posições, progressões, tratamento de vazões e extração de features de composição de gases ($	ext{Ar}$, $	ext{CO}_2$, $	ext{O}_2$, $	ext{N}_2$).
  * **Transformação de Features**: Padronização de variáveis numéricas (`StandardScaler`), codificação de categóricas (`OneHotEncoder`) e mapeamentos para camadas de *Embedding*.
* **Modelagem e Treinamento (`src/ml_model/`)**:

  * **Rede Neural Personalizada**: Arquitetura combinando vetores de *Embedding* para materiais com features numéricas/categóricas em blocos densos parametrizáveis (`Linear`, `ReLU`/`GELU`/`SiLU`, `BatchNorm1d` e `Dropout`).
  * **Validation & Loss Ponderada**: Suporte nativo a *K-Fold Cross Validation* e função de perda ajustada para priorizar variáveis críticas.
* **Otimização de Hiperparâmetros**:

  * Busca Bayesiana automatizada com **Optuna**  e exportação do modelo final performático para formato **ONNX**.

---

## 🚀 Como Executar

### 1. Instalação do Ambiente

Certifique-se de ter o Python instalado. Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # No Linux/macOS
# venv\Scripts\activate   # No Windows
pip install -r requirements.txt
```

### 2. Variáveis de Ambiente

Configure o arquivo `.env` na raiz do projeto com os caminhos dos seus dados:

```env
CAMINHO_DOS_DADOS=caminho/para/seu/dataset_bruto.csv
CAMINHO_DADOS_REESTRUTURADOS=caminho/para/seu/dataset_reestruturado.csv
DADOS_PROJETO=caminho/para/seu/dados_preprocessados.pt
```

### 3. Execução do Pipeline

```bash
# 1. Processamento e pré-processamento dos dados
python src/data_prep/load_data.py
python src/data_prep/preprocess_data.py
python src/data_prep/preprocess_encoder.py

# 2. Tuning de hiperparâmetros e treinamento final
python src/ml_model/tuning.py
```

---

## 📊 Métricas Avaliadas

O modelo monitora e otimiza individualmente e globalmente:

* **MSE** (*Mean Squared Error*)
* **MAE** (*Mean Absolute Error*)
* **R²** (*Coeficiente de Determinação*)
