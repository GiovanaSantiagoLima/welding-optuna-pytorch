# Otimização de Processos de Soldagem com Optuna e PyTorch

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-2563EB?style=flat&logo=optuna&logoColor=white)

Este repositório contém uma solução para a previsão e otimização de parâmetros no processo de soldagem, utilizando redes neurai com **PyTorch** e ajuste automatizado de hiperparâmetros através do **Optuna**.

O objetivo principal é encontrar a arquitetura de rede e os hiperparâmetros ideais para maximizar a precisão dos modelos preditivos responsáveis por determinar a **voltagem, amperagem e velocidade de soldagem** ideais para o processo.

---
## Funcionalidades

* **Modelagem Flexível:** Redes neurais customizáveis via PyTorch para regressão de variáveis de soldagem.
* **Otimização Inteligente:** Busca automatizada de hiperparâmetros (taxa de aprendizado, número de camadas, neurônios por camada, funções de ativação e otimizadores) utilizando amostragem TPE (*Tree-structured Parzen Estimator*) do Optuna.
* **Gestão de Experimentos:** Poda automatizada (*pruning*) de tentativas pouco promissoras para economizar tempo computacional.
* **Visualização de Resultados:** Geração de gráficos de importância de hiperparâmetros e histórico de otimização.
  
---
## Estrutura do Projeto 

├── data/               # Datasets de soldagem (CSVs, etc.)
├── models/             # Checkpoints dos melhores modelos salvos
├── notebooks/          # Notebooks Jupyer para análise exploratória
├── src/
│   ├── preprocess.py      # Pipeline de dados e DataLoader do PyTorch
│   └── train.py           # Script principal de otimização e treino
├── requirements.txt    # Dependências do projeto
└── README.md           # Documentação do projeto
