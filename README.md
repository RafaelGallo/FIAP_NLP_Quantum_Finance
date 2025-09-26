# Case QuantumFinance – Classificador de Chamados (Disciplina NLP)

## 📌 Contexto do Projeto

A **QuantumFinance**, empresa do setor financeiro digital, mantém um canal de atendimento via chat onde os clientes relatam dúvidas, solicitações e problemas.

O desafio é desenvolver um **modelo de inteligência artificial** capaz de **classificar automaticamente o assunto dos chamados** com base no texto livre fornecido pelo cliente, com o objetivo de:

* Otimizar o direcionamento das demandas.
* Reduzir tempo de atendimento.
* Aumentar a eficiência operacional.

## 🎯 Objetivos

* Desenvolver um **sistema de classificação de chamados** com técnicas de **Processamento de Linguagem Natural (PLN)**.
* Implementar e comparar **duas abordagens principais**:

  1. **TF-IDF + Modelos clássicos de Machine Learning**.
  2. **Embeddings (Word2Vec/BERTimbau) + Classificadores supervisionados**.
* Atingir **F1-Score ≥ 75%** no conjunto de teste.

## ⚙️ Requisitos Técnicos

* Pré-processamento textual (limpeza, normalização, lematização).
* Vetorização com **n-gramas + TF-IDF**.
* Extração de embeddings com **BERTimbau**.
* Implementação de pelo menos **dois pipelines completos**:

  * **Modelo tradicional (TF-IDF + ML)**.
  * **Modelo com embeddings + ML**.
* Avaliação com **F1-Score (weighted)**.

## 📊 Análises e Experimentos

### 🔹 Seleção de Hiperparâmetros (TF-IDF)

Testamos diferentes tamanhos de vocabulário no **TF-IDF** para identificar o melhor trade-off entre dimensionalidade e performance.

![TF-IDF Score](https://github.com/RafaelGallo/FIAP_NLP_Quantum_Finance/blob/main/img/01.png?raw=true)

### 🔹 Avaliação de Modelos Clássicos (TF-IDF)

| Modelo                 | Accuracy  | Precision | Recall    | F1-Score  |
| ---------------------- | --------- | --------- | --------- | --------- |
| DecisionTreeClassifier | 0.628     | 0.633     | 0.628     | 0.626     |
| RandomForestClassifier | 0.795     | 0.795     | 0.795     | 0.794     |
| LogisticRegression     | **0.891** | **0.892** | **0.891** | **0.891** |
| AdaBoostClassifier     | 0.800     | 0.799     | 0.800     | 0.799     |
| XGBClassifier          | 0.315     | 0.381     | 0.315     | 0.235     |
| LGBMClassifier         | 0.886     | 0.886     | 0.886     | 0.886     |

🔑 **Conclusão**:

* **Logistic Regression (TF-IDF)** foi o **melhor modelo** com F1 = **0.891**.
* **LightGBM (TF-IDF)** apresentou desempenho muito próximo.
* **RandomForest** e **AdaBoost** foram alternativas estáveis (~0.79 F1).
* **XGBoost** teve desempenho insatisfatório (F1 = 0.23).

### 🔹 Matrizes de Confusão (TF-IDF e Transformers)

* **Logistic Regression (TF-IDF)**
  ![Confusão LR](https://github.com/RafaelGallo/FIAP_NLP_Quantum_Finance/blob/main/img/35.png?raw=true)

### 🔹 Curvas ROC Multiclasse

* **Logistic Regression (TF-IDF)**
  ![ROC LR](https://github.com/RafaelGallo/FIAP_NLP_Quantum_Finance/blob/main/img/36.png?raw=true)

* **BERTimbau + ML (RandomForest, AdaBoost, LogisticRegression)**
  ![ROC BERT](https://github.com/RafaelGallo/FIAP_NLP_Quantum_Finance/blob/main/img/05.png?raw=true)

Boa, Rafael 👏
Essa imagem é essencial para explicar a base teórica do projeto (Transformers → BERTimbau). O ideal é incluir no README uma seção dedicada à **Arquitetura Transformer**, contextualizando o uso do encoder e a diferença em relação ao modelo seq2seq tradicional.

## 🏗️ Arquitetura Transformer

O modelo **Transformer** (Vaswani et al., 2017) é a base para arquiteturas modernas de PLN, incluindo o **BERTimbau** utilizado neste projeto.

Ele é composto por duas partes principais:

* **Encoder**: responsável por processar o texto de entrada e gerar representações contextuais (embeddings enriquecidos).
* **Decoder**: usado em tarefas de geração de texto (não utilizado no BERT, que emprega apenas a parte Encoder).

O mecanismo central é o **Self-Attention**, que permite ao modelo capturar dependências de longo alcance entre palavras em uma frase.

![Arquitetura Transformer](https://github.com/RafaelGallo/FIAP_NLP_Quantum_Finance/blob/main/img/03.png?raw=true)

🔑 **Resumo aplicado ao projeto**:

* Utilizamos o **Encoder do BERTimbau** para gerar embeddings em português.
* Esses embeddings foram usados diretamente em classificadores supervisionados (Logistic Regression, RandomForest, AdaBoost etc.).
* O uso do Transformer permitiu capturar **relações semânticas contextuais** entre termos dos chamados, indo além da simples contagem de palavras (TF-IDF).

## 🧩 Transformers – BERTimbau

Utilizamos o modelo **BERTimbau (NeuralMind)**, pré-treinado para português.
Dois cenários foram testados:

1. **Fine-tuning direto (classificação supervisionada)**.
2. **Extração de embeddings** para alimentar classificadores como Logistic Regression, Random Forest, AdaBoost, etc.

Apesar da riqueza semântica dos embeddings, os resultados **não superaram o TF-IDF** neste problema específico.

## 📉 Comparação Final – TF-IDF vs Embeddings

| Modelo                            | F1-Score  |
| --------------------------------- | --------- |
| LogisticRegression (TF-IDF)       | **0.891** |
| LightGBM (TF-IDF)                 | 0.886     |
| RandomForest (TF-IDF)             | 0.794     |
| AdaBoost (TF-IDF)                 | 0.799     |
| LogisticRegression (Transformers) | 0.738     |
| XGBoost (Transformers)            | 0.701     |
| LGBM (Transformers)               | 0.684     |
| DecisionTree (Transformers)       | 0.415     |

✅ **TF-IDF + Logistic Regression continua sendo a melhor abordagem neste cenário.**

## 🚨 Leakage (Vazamento de Dados)

* Investigamos potenciais **colunas derivadas incorretamente do target**.
* Foram removidas features que poderiam causar **vazamento de informação** (ex: colunas calculadas antes do `train_test_split`).
* Isso garantiu que o modelo só tivesse acesso a variáveis **disponíveis em produção**.

## 🔮 Conclusões Finais

1. **Modelos clássicos com TF-IDF superaram embeddings BERT** neste caso.
2. **Logistic Regression (TF-IDF)** foi o **modelo campeão**, atingindo F1-Score de **0.891**.
3. **LightGBM** é uma ótima alternativa, embora apresente sinais leves de overfitting.
4. Embeddings BERT + ML não performaram tão bem sem fine-tuning completo.

## 🚀 Próximos Passos

* Realizar **fine-tuning supervisionado completo do BERTimbau**.
* Testar **Sentence-BERT** e outros embeddings contextuais.
* Aplicar **Explainable AI (LIME/SHAP)** para entender melhor as decisões dos modelos.
* Avaliar **pipelines de ensemble** (LR + LGBM).

## 📂 Estrutura do Repositório

```
├── data/                   # Conjunto de dados (não incluso no repositório público)
├── img/                    # Gráficos e matrizes de confusão
├── notebooks/
│   ├── Template_Trabalho_Final_NLP.ipynb
│   └── Experimentos_Modelos.ipynb
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```
