# Resumo Técnico: Comparativo Metodológico e Guia de Modelação (Capítulo 10 vs. Capítulo 11 V3)

## 1. Visão Geral do Problema
* **Objetivo:** Prever a matriz de faturação ($Y$) com base nos dados do CRM ($X$).
* **Natureza dos Dados:** Dataset de grande dimensão, inteiramente composto por **variáveis binárias** ($0$s e $1$s), de origem académica (variáveis anónimas).
* **Tipo de Problema:** **Classificação Multi-Rótulo (*Multi-Label Classification*)**, pois o alvo $Y$ é constituído por 731 variáveis binárias inter-relacionadas (algumas das quais aditivas/subtotais).

---

## 2. Comparativo Detalhado: Capítulo 10 (Baseline V1) vs. Capítulo 11 (Auditado V3)

| Dimensão / Etapa | Capítulo 10 (Baseline Original) | Capítulo 11 (V3 Auditada) | Justificação Técnica e Impacto Prático |
| :--- | :--- | :--- | :--- |
| **Filosofia do Target ($Y$)** | Trata as 731 colunas em bloco; aplica **relabeling forçado** por voto maioritário ($\ge 65\%$). | **Paradigma estritamente Multi-Label**; decompõe $Y$ em 606 alvos de ML e 125 regras aditivas. Preserva dados reais. | Elimina o risco de apagar variações legítimas ($\ge 3$ bits de distância Hamming) e garante 100% de precisão determinística nas regras aditivas. |
| **Filtro de Variância em $X$** | Remoção estrita de variância zero ($2$ colunas 100% nulas descartadas). | **Low Variance Filter Assintótico** ($n \cdot p \ge 5$, critério de Cochran). | Elimina variáveis ultra-raras ($< 5$ ativações em 187k linhas) que provocavam overfitting e instabilidade estatística. |
| **Seleção de Features em $X$** | Colapso por correlação ($\vert{}r\vert{} \ge 0.95$) com descarte por **ordem alfabética** do nome da coluna. | **$\chi^2$ vetorizado $X \times Y$ com controlo de FDR (Benjamini-Hochberg, $q=0.05$)** + Feature Union. | Filtra preditores irrelevantes com base na significância estatística real contra o target, em vez de cortes ingénuos por nome. |
| **TruncatedSVD ($X$)** | Nenhuma redução dimensional aplicada. | **TruncatedSVD Dinâmico ($\ge 80\%$ de variância acumulada)** $\to k^* \approx 45$ componentes. | Extrai representações latentes densas para modelos de árvore sem sofrer de data leakage (`fit` no treino, `transform` no teste). |
| **Complexidade de ML** | O modelo de ML tem de prever todas as **731 targets**. | O modelo prevê **606 targets**; as 125 aditivas são resolvidas por pós-processamento lógico. | Redução de 20% no espaço de hipóteses, acelerando o treino e aumentando o Exact Match Ratio (EMR). |
| **Nomenclatura de Ficheiros** | `train_clean.csv`, `test_clean.csv` | `train_clean_v2.csv`, `test_clean_v2.csv`, `train_targets_v2.csv` | Permite testes A/B diretos e isolados no Sprint 2 sem sobreposição de ficheiros. |

---

## 3. Análise das Duas Abordagens de TruncatedSVD e do Scree Plot

### A. Diferença entre o SVD da Secção 7a e da Secção 11.5
1. **SVD Não-Supervisionado para Clustering (Secção 7a — Rejeitado):**
   * *Problema:* Em matrizes Bernoulli sem centração na média (`with_mean=False`), a 1.ª componente singular absorve a norma/volume global de pacotes do cliente (tamanho do cliente). Como a variância de um pacote frequente ($p=0.5$) é até 180 vezes superior à de um pacote raro ($p=0.002$), o SVD ignora a cauda longa de serviços. Usá-lo para criar *clusters* rígidos destruiria os pacotes raros de faturação.
2. **SVD Híbrido Supervisionado (Secção 11.5 — Aprovado):**
   * *Solução:* O SVD **não substitui** as variáveis originais. As componentes contínuas são combinadas (*Feature Union*) com as variáveis binárias exatas selecionadas pelo $\chi^2$ (FDR). Fornece aos modelos de árvore um sinal contínuo do perfil macro do cliente sem perder a precisão dos pacotes raros.

### B. Diagnóstico do Pico na 2.ª Componente do Scree Plot
* **Pico na Componente 2 ($\approx 30.5\%$ de variância explicada):** 
  Assim que a 1.ª componente absorve o vetor de escala/densidade geral de consumo, a 2.ª componente alinha-se com o **maior eixo de separação do catálogo de CRM** (ex.: a grande divisão entre *Clientes Móvel Puro* vs. *Clientes Pacotes Fixo/TV*).
* **Conexão com a Rejeição da Secção 7a:** 
  O gráfico demonstra que 2 componentes concentram $\sim 35\%$ da variância total, caindo vertiginosamente a partir da 4.ª componente ($< 2\%$ cada). Isto prova visualmente que o SVD concentra a informação em poucos eixos "macro". Para *clustering*, isto é nocivo (esconde a cauda longa); para *enriquecimento de features* (Secção 11.5), é ideal.

---

## 4. Guia de Modelação para o Teste A/B (Sprint 2)

### A. Estratégia para o Capítulo 10 (Baseline V1)
Nesta abordagem, o modelo lida diretamente com as 731 colunas de faturação (com rótulos alterados por relabeling):
* **Modelos Recomendados:**
  1. **LightGBM / XGBoost com `MultiOutputClassifier` (ou Binary Relevance):** Treino de 731 árvores independentes para servir de benchmark de comparação.
  2. **Regressão Logística Regularizada (L1 / Lasso ou L2 / Ridge):** Benchmark estatístico rápido. Como o relabeling reduziu o ruído, serve como limite mínimo de desempenho.
  3. **Classifier Chains (Cadeias de Classificadores):** Permite encadear previsões para capturar dependências entre as 731 variáveis de faturação.
  4. **Bernoulli Naive Bayes:** Benchmark de velocidade extrema para dados totalmente binários.

### B. Estratégia para o Capítulo 11 (V3 Auditada)
Nesta abordagem, o modelo prevê **606 alvos de ML** com o dataset enriquecido ($\chi^2$ FDR + $k^* \approx 45$ SVD) e utiliza pós-processamento determinístico:
* **Modelos Recomendados:**
  1. **LightGBM com `MultiOutputClassifier`:** Otimizado para esparsidade, velocidade e baixo consumo de RAM.
  2. **CatBoost (`CatBoostClassifier`):** Excelente na captura de interações complexas entre variáveis binárias/contínuas sem overfitting.
  3. **XGBoost (`XGBClassifier` com `tree_method='hist'`):** Elevada performance e paralelização eficiente em matrizes esparsas.
  4. **Rede Neuronal Multi-Saída (MLP Deep Learning):** Arquitetura com camadas densas, Dropout e uma camada de saída de **606 neurónios com ativação `sigmoid`** e perda `binary_crossentropy`. Aprende representações partilhadas num único modelo.

### C. Fluxo de Pós-Processamento e Avaliação no Capítulo 11

[CRM Test (X_test)] ──► Model ML (606 alvos) ──► Previsões Probabilísticas (606)
│
▼
[Additive Rules JSON] ──► reconstruct_full_predictions() ◄── [Ativações CRM]
│
▼
Previsão Final (731 alvos)
│
▼
evaluate_multilabel_performance()
(ROC-AUC, F1-Score, Exact Match Ratio)


1. O modelo de ML gera as previsões para os 606 alvos base.
2. A função `reconstruct_full_predictions` avalia os gatilhos lógicos de CRM e recria as **125 colunas aditivas** com 100% de precisão teórica.
3. A matriz final de 731 colunas reconstruídas é avaliada contra o conjunto de teste real via **ROC-AUC (Micro/Macro)**, **F1-Score (Micro/Macro)** e **Exact Match Ratio (EMR)**.