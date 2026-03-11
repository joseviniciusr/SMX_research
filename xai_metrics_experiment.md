# XAI Metrics Experiment Protocol

## Goal
Define a reproducible evaluation protocol for SMX as an XAI method on spectral data, with direct comparison against SHAP, VIP, and a simple spectral baseline (permutation importance by zone).
The protocol combines objective and domain-grounded evaluation dimensions [R1].

## Scope
- Task type: binary classification (main metric: F1 and Accuracy).
- Explanation granularity:
  - Zone-level ranking.
- Methods:
  - `SMX`
  - `SHAP`
  - `VIP`
  - `Permutation-by-zone` (simple spectral baseline)

## Common Protocol (for all experiments)

### Data splitting
- Use `10` split seeds (`split_seed`).
- For each `split_seed`:
  - Shuffle data and run repeated stratified CV.
  - Recommended: `5x2` repeated stratified K-fold (5 folds, 2 repeats).
- Keep train/test splits identical across explainers within each seed/fold.

### Seed control
- `split_seed`: controls only dataset shuffling and train/test fold assignment.
- `model_seed`: controls only model creation/training randomness (e.g., MLP initialization, stochastic optimizer order).
- Use both and log both in outputs.
- Recommended protocol:
  - Stability wrt data split: vary `split_seed`, keep `model_seed` fixed.
  - Stability wrt model stochasticity: vary `model_seed`, keep `split_seed` fixed.
  - Full variability analysis: vary both.

### Models
- Compare at least: `PLS-DA`, `SVM`, `MLP`.
- Fit model on train split only.
- Compute test predictions and test performance (F1, Accuracy).

### Explanation extraction
- Always compute explanations on test samples (or train+test if explicitly needed, but report which one).
- For each method, export:
  - Zone ranking list.

### Statistical reporting
- Report mean and `95% CI` over all seed-fold runs.
- CI method: bootstrap CI on run-level scores (recommended 2000 resamples).
- Paired test (method vs method on same runs):
  - If paired differences are approximately normal: paired t-test.
  - Otherwise: Wilcoxon signed-rank test.
- Report effect size where possible.
  Statistical references: [R12], [R13], [R14].

---

## 1) Faithfulness

Faithfulness here means: if top-ranked explanation units are removed, model performance should degrade more.
This is aligned with perturbation-based faithfulness/comprehensiveness evaluation [R5].

### 1.1 Sub-experiment A: Rank zones and remove top-k zones

#### Ranking definitions
- SMX: use zone importance from SMX relevance (e.g., LRC-based natural ranking).
- SHAP: zone score = mean absolute SHAP value aggregated per zone [R6].
- VIP: zone score = mean VIP per zone [R7].
- Permutation baseline: zone score = permutation importance per zone.

#### Removal operator
- For each method and each `k` in `{1, 2, 3, 5, 10, 20, ...}` (bounded by number of zones):
  - Remove (mask) top-k zones in test data.
  - Apply the same masking rule for all methods.
  - Re-evaluate model with no retraining.

#### Metrics
- `F1_drop(k) = F1_original - F1_masked(k)`
- `ACC_drop(k) = ACC_original - ACC_masked(k)`
- Optional summary: area under degradation curve (larger means more faithful ranking).


---

## 2) Stability

Measure ranking consistency using RBO under controlled seed variation.
This uses RBO for top-weighted rank similarity [R9].

### Definition
- For each method, build ranking lists from repeated runs with explicit seed mode:
  - `split-seed stability`: rankings across different `split_seed` with fixed `model_seed`.
  - `model-seed stability`: rankings across different `model_seed` with fixed `split_seed`.
- Compute pairwise `RBO` among runs of the same method.
- Stability score = mean pairwise RBO.

### Report
- `mean_RBO_zones_split_seed(method)`
- `mean_RBO_zones_model_seed(method)`
- Include 95% CI.

Recommended RBO parameter:
- `p = 0.7` (top-weighted; adjust if stronger top emphasis is desired).

---

## 3) Model Dependence

Assess how explanation rankings depend on model family.
Motivation: explanations can vary across different model classes fitted on the same data (Rashomon/model dependence) [R10], [R11].

### Procedure
- Reuse the same seed/fold protocol from Stability.
- For each method (SMX, SHAP, VIP, permutation):
  - Compare rankings generated from different algorithms (`PLS-DA`, `SVM`, `MLP`) on the same data splits.
  - Use fixed `split_seed` and fixed `model_seed` when comparing algorithms in the same run.
  - Compute RBO between algorithm pairs:
    - `PLS-DA vs SVM`
    - `PLS-DA vs MLP`
    - `SVM vs MLP`

### Report
- `model_dependence_RBO` per method and per pair.
- Lower RBO => stronger model dependence.

---

## 4) Domain-Grounded Evaluation

Human expert validation of top zones.
This corresponds to application-grounded evaluation [R1], [R12].

### Input to expert
- For each method and run, show top-N zones with:
  - mapped spectral range,
  - associated elemental annotation (if available).

### Expert task
- For each item: mark `physically_plausible` as `yes/no`.

### Metrics
- Expert agreement rate:
  - `agreement_rate = (# yes) / (total reviewed)`
---

## 5) Baselines and Comparison Matrix

Methods to include:
- SMX
- SHAP
- VIP
- Permutation-by-zone baseline

Compare methods on:
- Faithfulness (zone only)
- Stability (RBO across seeds)
- Model dependence (RBO across algorithms)
- Domain-grounded agreement
- Comprehensibility (section 6)

All comparisons must use:
- identical seed/fold splits,
- identical metric computation.

---

## 6) Comprehensibility

Focus metrics:

### 6.1 Number of top zones to explain 80% relevance
- Sort zones by relevance descending.
- Compute cumulative relevance ratio.
- `N80_zones = minimum n such that cumulative_relevance(n) >= 0.80`.
- Lower is more comprehensible (more compact explanation).
Related interpretability notions: simulatability and compactness [R15], [R16].

### 6.2 Percentage of top-ranked zones matching expected elemental signatures
- Define expected elemental signature zones from domain knowledge.
- For top-K zones:
  - `elemental_match_pct = (# top-K zones in expected list) / K`.
- Report at K = 2, K = 5 and K = 10.

---

## Minimal Output Artifacts

Recommended CSV outputs:
- `faithfulness_zone_curve.csv`
  - columns: `split_seed, model_seed, fold, model, method, k, f1_original, f1_masked, f1_drop, acc_original, acc_masked, acc_drop`
- `stability_rbo.csv`
  - columns: `method, granularity(zone), stability_mode(split_seed|model_seed), run_i, run_j, split_seed_i, model_seed_i, split_seed_j, model_seed_j, rbo`
- `model_dependence_rbo.csv`
  - columns: `method, granularity(zone), model_a, model_b, split_seed, model_seed, fold, rbo`
- `domain_grounded_expert_review.csv`
  - columns: `split_seed, model_seed, method, model, item_type(zone), item_id, rank, plausible_yes_no, expert_id`
- `comprehensibility_metrics.csv`
  - columns: `split_seed, model_seed, fold, model, method, n80_zones, elemental_match_k5, elemental_match_k10`

Recommended summary file:
- `xai_metrics_summary.csv`
  - mean, 95% CI, paired-test p-values for each metric and method comparison.

---

## References

- [R1] Vilone, G., & Longo, L. (2021). *Notions of explainability and evaluation approaches for explainable artificial intelligence*. Information Fusion, 76, 89-106.
- [R5] Yeh, C.-K., Kim, J. S., Yen, I. E.-H., & Ravikumar, P. (2019). *On the (In)fidelity and Sensitivity of Explanations*. NeurIPS.
- [R6] Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- [R7] Chong, I.-G., & Jun, C.-H. (2005). *Performance of some variable selection methods when multicollinearity is present*. Chemometrics and Intelligent Laboratory Systems, 78(1-2), 103-112.
- [R9] Webber, W., Moffat, A., & Zobel, J. (2010). *A Similarity Measure for Indefinite Rankings*. ACM TOIS, 28(4), 20.
- [R10] Breiman, L. (2001). *Statistical Modeling: The Two Cultures*. Statistical Science, 16(3), 199-231.
- [R11] Rudin, C. (2019). *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead*. Nature Machine Intelligence, 1, 206-215.
- [R12] Doshi-Velez, F., & Kim, B. (2017). *Towards A Rigorous Science of Interpretable Machine Learning*. arXiv:1702.08608.
- [R13] Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
- [R14] Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*. Neural Computation, 10(7), 1895-1923.
- [R15] Lipton, Z. C. (2016). *The Mythos of Model Interpretability*. arXiv:1606.03490.
- [R16] Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.).

---

# Protocolo de Avaliação de XAI para SMX — Explicação Teórica Detalhada

---

## 0. Preliminares: Como o SMX funciona (recapitulação rápida)

Para entender as métricas de avaliação, é preciso primeiro entender o que o SMX produz como explicação. O pipeline é:

1. **Extração de zonas**: O espectro (e.g., XRF) é particionado em $Z$ zonas espectrais. Cada zona $z_j$ é um trecho contíguo do espectro (e.g., "Ca ka: 3.6–3.7 keV").

2. **Agregação por PCA**: Para cada zona, ajusta-se um PCA de 1 componente. Cada amostra $i$ recebe um score escalar por zona:
$$s_{ij} = \mathbf{l}_j^\top (\mathbf{x}_{ij} - \bar{\mathbf{x}}_j)$$
onde $\mathbf{l}_j$ são os loadings da PC1 da zona $j$, e $\bar{\mathbf{x}}_j$ é a média ajustada no treino.

3. **Geração de predicados**: Criam-se regras binárias tipo "$s_j \leq q$" ou "$s_j > q$" para quantis $q \in \{0.2, 0.4, 0.6, 0.8\}$.

4. **Bagging**: Fazem-se reamostragens (bags) de amostras e predicados para robustez.

5. **Métricas de predicado**: Dentro de cada bag, mede-se a importância de cada predicado por **covariância** ($|\text{Cov}(s_j, \hat{y})|$) ou **perturbação** (zerar a zona e ver o impacto na predição).

6. **Grafo dirigido**: Constrói-se um grafo onde predicados são nós e arestas representam dependências ponderadas pelo ranking das métricas.

7. **LRC (Local Reaching Centrality)**: Calcula-se a centralidade de cada nó no grafo. Predicados com **LRC alto** são os mais "importantes" — eles alcançam mais nós, indicando influência estrutural.

8. **Ranking de zonas**: Agrega-se o LRC por zona (deduplicação: uma zona pode ter vários predicados, mas só a de maior LRC é mantida), obtendo o **ranking final de zonas** do SMX.

**Output principal do SMX**: Uma lista ordenada de zonas espectrais, da mais relevante (maior LRC) para a menos relevante.

---

## 1. Faithfulness (Fidelidade)

### O que é
Fidelidade responde à pergunta: **"O que o método diz ser importante realmente importa para o modelo?"** Se as zonas mais bem ranqueadas forem removidas e a performance cair muito, a explicação é **fiel** ao modelo.

### Fundamento teórico
Baseia-se no conceito de **comprehensiveness** (abrangência) de Yeh et al. (2019) [R5]: uma explicação é fiel se remover os features mais importantes degrada substancialmente as predições.

### Procedimento detalhado

#### Passo 1: Obter o ranking de zonas de cada método

Para cada método $m \in \{\text{SMX}, \text{SHAP}, \text{VIP}, \text{Permutation}\}$, obtém-se um ranking $\mathcal{R}_m = [z_{(1)}, z_{(2)}, \ldots, z_{(Z)}]$ onde $z_{(1)}$ é a zona mais importante.

**Como cada método gera o ranking:**

- **SMX**: $\text{score}(z_j) = \text{LRC}_{\text{agregado}}(z_j)$. É o LRC médio sobre seeds do predicado de maior LRC para aquela zona. Já existe no output `lrc_pert_natural.csv`.

- **SHAP**: Para cada feature (canal de energia), calcula-se $\phi_i^{(k)}$ (valor SHAP da feature $k$ para amostra $i$). Agrega-se por zona:
$$\text{score}_{\text{SHAP}}(z_j) = \frac{1}{n} \sum_{i=1}^{n} \sum_{k \in z_j} |\phi_i^{(k)}|$$
ou seja, média sobre amostras da soma dos SHAPs absolutos dos canais pertencentes à zona $j$.

- **VIP** (só para PLS): Variable Importance in Projection. Para o canal $k$:
$$\text{VIP}_k = \sqrt{p \cdot \frac{\sum_{a=1}^{A} w_{ak}^2 \cdot SS_a}{\sum_{a=1}^{A} SS_a}}$$
onde $p$ é o número de variáveis, $w_{ak}$ é o peso do canal $k$ no componente latente $a$, $SS_a$ é a soma de quadrados explicada pelo componente $a$, e $A$ é o número de componentes. Agrega-se por zona: $\text{score}_{\text{VIP}}(z_j) = \text{mean}_{k \in z_j}(\text{VIP}_k)$.

- **Permutation importance por zona**: Permuta-se aleatoriamente todos os canais pertencentes à zona $j$ e mede-se a queda de desempenho:
$$\text{score}_{\text{perm}}(z_j) = F_1^{\text{original}} - F_1^{\text{permuted}(z_j)}$$

#### Passo 2: Remoção progressiva (top-k zones masking)

Para cada $k \in \{1, 2, 3, 5, 10, 20, \ldots\}$ (limitado pelo número total de zonas):

1. Toma-se as **top-k zonas** segundo o ranking do método $m$.
2. No conjunto de **teste**, substitui-se (mascara) os canais dessas $k$ zonas por um valor neutro (tipicamente zero ou a mediana da zona no treino).
3. **Sem retreinar o modelo**, aplica-se o modelo treinado nos dados mascarados.
4. Calcula-se a performance no teste mascarado.

#### Passo 3: Métricas de fidelidade

Para cada $k$:

$$F_1\text{-drop}(k) = F_1^{\text{original}} - F_1^{\text{masked}(k)}$$

$$\text{ACC-drop}(k) = \text{ACC}^{\text{original}} - \text{ACC}^{\text{masked}(k)}$$

**Resumo via área sob a curva de degradação (AUC-degradation)**:

$$\text{AUDC}(m) = \sum_{k} F_1\text{-drop}(k) \cdot \Delta k$$

onde $\Delta k$ é o incremento no eixo $x$. Quanto **maior** a AUDC, **mais fiel** é o ranking: significa que remover as zonas top causou grande degradação.

#### Exemplo concreto

Suponha um dataset com 10 zonas e um MLP treinado com $F_1 = 0.95$.

| $k$ | SMX $F_1^{\text{masked}}$ | SHAP $F_1^{\text{masked}}$ |
|-----|---------------------------|----------------------------|
| 1   | 0.80                      | 0.88                       |
| 2   | 0.65                      | 0.72                       |
| 3   | 0.50                      | 0.60                       |
| 5   | 0.40                      | 0.48                       |

Aqui, SMX causa maior queda → $F_1\text{-drop}_{\text{SMX}}(1) = 0.15$ vs. $F_1\text{-drop}_{\text{SHAP}}(1) = 0.07$. O SMX seria **mais fiel** neste cenário.

#### O que esperar
- **Métodos fiéis** produzem curvas de degradação íngremes (queda rápida de F1 com poucos $k$).
- **Métodos fracos** produzem curvas planas ou até inversas (a remoção melhora a performance, indicando que o ranking não captura o que o modelo usa).
- Se todos os métodos têm AUDC similar, a fidelidade não diferencia os métodos nesse dataset.

---

## 2. Stability (Estabilidade)

### O que é
Estabilidade responde: **"Se eu repetir o procedimento em condições ligeiramente diferentes, obtenho explicações parecidas?"** Uma explicação instável não é confiável — muda a cada execução.

### Fundamento teórico
Usa-se **RBO (Rank-Biased Overlap)** de Webber et al. (2010) [R9], uma medida de similaridade entre rankings que dá **mais peso ao topo** do ranking.

### Definição formal do RBO

Sejam duas listas ranqueadas $S$ e $T$, e defina $A_d$ e $B_d$ como os conjuntos dos $d$ primeiros itens de $S$ e $T$ respectivamente. O **agreement at depth $d$** é:

$$X_d = \frac{|A_d \cap B_d|}{d}$$

O RBO com parâmetro de convergência $p \in (0,1)$ é:

$$\text{RBO}(S, T, p) = (1 - p) \sum_{d=1}^{\infty} p^{d-1} \cdot X_d$$

Na prática, calcula-se com uma fórmula truncada (extrapolada) no comprimento finito dos rankings. O parâmetro $p$ controla o viés para o topo:
- $p = 0.7$ (recomendado no protocolo): as **3–4 primeiras posições** dominam o score.
- $p = 0.9$: mais posições do ranking contribuem.
- $p = 0.5$: praticamente só a primeira posição importa.

**Interpretação**: RBO = 1 → rankings idênticos; RBO = 0 → rankings completamente disjuntos no topo.

### Dois modos de estabilidade

#### Modo A: Estabilidade ao split de dados (`split-seed stability`)

**O que varia**: o seed que controla a partição treino/teste (`split_seed`).  
**O que é fixo**: o seed de inicialização do modelo (`model_seed`).

**Procedimento**:
1. Para cada `split_seed` $\in \{s_1, s_2, \ldots, s_{10}\}$, com `model_seed` fixo:
   - Treinar modelo, obter ranking de zonas pelo método $m$.
   - Resultado: 10 rankings $\mathcal{R}_m^{(s_1)}, \mathcal{R}_m^{(s_2)}, \ldots$
2. Calcular **todos os pares** de RBO:
$$\text{RBO}_{ij} = \text{RBO}(\mathcal{R}_m^{(s_i)}, \mathcal{R}_m^{(s_j)}, p=0.7), \quad \forall i < j$$
3. Score de estabilidade:
$$\text{Stability}_{\text{split}}(m) = \frac{2}{n(n-1)} \sum_{i<j} \text{RBO}_{ij}$$

**O que mede**: se os dados mudam ligeiramente (amostras diferentes no treino/teste), o ranking permanece? Se sim, a explicação é **robusta à variação amostral**.

#### Modo B: Estabilidade ao modelo (`model-seed stability`)

**O que varia**: o seed de inicialização do modelo (`model_seed`).  
**O que é fixo**: `split_seed`.

Procedimento idêntico, mas agora varia-se a aleatoriedade do modelo (e.g., inicialização dos pesos do MLP, ordem de otimização do SVM). Para PLS, que é determinístico, espera-se estabilidade perfeita (RBO ≈ 1).

#### Exemplo concreto

Suponha 3 seeds para simplificar, e 5 zonas {Ca, Fe, Ti, Si, Mg}:

| Seed | SMX Ranking         | SHAP Ranking        |
|------|---------------------|---------------------|
| 0    | Ca > Fe > Ti > Si > Mg | Fe > Ca > Ti > Si > Mg |
| 1    | Ca > Fe > Si > Ti > Mg | Ti > Fe > Ca > Mg > Si |
| 2    | Ca > Ti > Fe > Si > Mg | Fe > Ti > Ca > Si > Mg |

Pares RBO (p=0.7) para SMX:
- (0,1): Ca=Ca (posição 1 igual), Fe vs Fe em posições 2 vs 2 → alto RBO ≈ 0.85
- (0,2): Ca=Ca, Fe vs Ti trocam → RBO ≈ 0.80
- (1,2): Ca=Ca, Fe e Ti trocam → RBO ≈ 0.78

$\text{Stability}_{\text{split}}(\text{SMX}) \approx 0.81$

Para SHAP: posição 1 muda entre Fe e Ti → RBO menor → $\text{Stability}_{\text{split}}(\text{SHAP}) \approx 0.55$

**Conclusão**: SMX mais estável que SHAP neste caso.

#### O que esperar
- **PLS + VIP**: altamente estável (determinístico), RBO próximo de 1.
- **MLP + SHAP**: pode ser instável (inicialização aleatória propaga para SHAP).
- **SMX**: a estabilidade depende do bagging interno e do LRC. O bagging com múltiplos seeds internos tende a **estabilizar** o SMX.
- Reportar com **IC 95%** (via bootstrap) sobre os scores RBO.

---

## 3. Model Dependence (Dependência do Modelo)

### O que é
Responde: **"O ranking de importância muda quando trocamos a familia de modelo (PLS vs SVM vs MLP)?"** Se muda muito, a explicação é uma propriedade do modelo, não dos dados. Se muda pouco, as explicações são mais universais.

### Fundamento teórico
Ligado ao **efeito Rashomon** (Breiman, 2001 [R10]): múltiplos modelos podem ter performance similar mas diferir na forma como usam os features. Rudin (2019) [R11] argumenta que métodos XAI pós-hoc podem gerar explicações inconsistentes entre modelos.

### Procedimento

1. Fixar `split_seed` e `model_seed` (mesmos dados, mesma aleatoriedade).
2. Para cada método $m$ (SMX, SHAP, VIP, Permutation):
   - Treinar PLS-DA, SVM, MLP nos **mesmos dados**.
   - Obter ranking do método $m$ com cada modelo: $\mathcal{R}_m^{\text{PLS}}, \mathcal{R}_m^{\text{SVM}}, \mathcal{R}_m^{\text{MLP}}$.
3. Calcular RBO entre pares de modelos:

$$\text{MD}_{\text{PLS,SVM}}(m) = \text{RBO}(\mathcal{R}_m^{\text{PLS}}, \mathcal{R}_m^{\text{SVM}}, p=0.7)$$

$$\text{MD}_{\text{PLS,MLP}}(m) = \text{RBO}(\mathcal{R}_m^{\text{PLS}}, \mathcal{R}_m^{\text{MLP}}, p=0.7)$$

$$\text{MD}_{\text{SVM,MLP}}(m) = \text{RBO}(\mathcal{R}_m^{\text{SVM}}, \mathcal{R}_m^{\text{MLP}}, p=0.7)$$

4. Repetir sobre múltiplos `split_seed` e reportar média ± IC 95%.

### Interpretação

| $\text{RBO}$ | Significado |
|---------------|-------------|
| > 0.8         | Explicação **model-agnostic** de facto — rankings consistentes entre modelos |
| 0.5 – 0.8    | Dependência moderada — top zones similar mas ordem varia |
| < 0.5         | Forte dependência do modelo — explicações contradizem-se |

### Exemplo

| Par de modelos | SMX RBO | SHAP RBO | Permutation RBO |
|----------------|---------|----------|-----------------|
| PLS vs SVM     | 0.72    | 0.45     | 0.60            |
| PLS vs MLP     | 0.68    | 0.40     | 0.55            |
| SVM vs MLP     | 0.85    | 0.70     | 0.75            |

**Interpretação**: SMX e Permutation são relativamente model-agnostic. SHAP varia mais entre modelos — captura idiossincrasias de cada modelo (o que não é necessariamente ruim, mas dificulta a interpretação universal).

### Nota sobre VIP
VIP só existe para PLS, então **não há como calcular model dependence do VIP** entre modelos. VIP entra apenas como comparador dentro do PLS.

### O que esperar
- **SMX** tende a ser moderadamente estável entre modelos porque o pipeline (zonas → PCA → predicados → grafo) é estruturalmente o mesmo, mas o input muda (predições do modelo).
- **SHAP** pode ser muito sensível ao modelo porque é calculado diretamente a partir das predições individuais.
- Quanto mais os modelos concordam na performance, mais seus rankings devem concordar (se o modelo captura o mesmo sinal dos dados).

---

## 4. Domain-Grounded Evaluation (Avaliação por Especialista)

### O que é
É a validação humana: **"As zonas top fazem sentido físico/químico?"** Um método pode ser fiel ao modelo mas apontar zonas sem significado real.

### Fundamento teórico
Corresponde à avaliação **application-grounded** de Doshi-Velez & Kim (2017) [R12] e às noções de Vilone & Longo (2021) [R1]: a utilidade real de uma explicação depende do julgamento de especialistas no domínio.

### Procedimento

1. Para cada método $m$, modelo e run:
   - Extrair as **top-N zonas** do ranking.
   - Mapear cada zona para sua faixa espectral real (e.g., "Ca Kα: 3.69 keV").
   - Se possível, associar à **assinatura elemental** conhecida (e.g., "Ca Kα" corresponde a Cálcio).

2. **Apresentar ao especialista** (ou grupo de especialistas), sem revelar qual método gerou o ranking:
   - Para cada item: "Zona X (faixa 3.6–3.7 keV, associada a Ca). Esta zona é **fisicamente plausível** como discriminante entre as classes A e B?"
   - O especialista marca: **sim/não**.

3. **Calcular métricas**:
$$\text{agreement\_rate}(m) = \frac{\#\text{sim}}{\#\text{total revisados}}$$

### Exemplo

Para o dataset `bank_notes` com classes A e B diferenciadas por composição elementar:

| Rank | SMX top zone | Plausível? | SHAP top zone | Plausível? |
|------|-------------|------------|---------------|------------|
| 1    | Ca ka       | Sim        | Fe ka         | Sim        |
| 2    | Ti ka       | Sim        | background    | Não        |
| 3    | Fe ka       | Sim        | Ti ka         | Sim        |
| 4    | Si ka       | ?          | noise region  | Não        |
| 5    | Mg ka       | Sim        | Ca ka         | Sim        |

$\text{agreement\_rate}(\text{SMX}) = 4/5 = 0.80$  
$\text{agreement\_rate}(\text{SHAP}) = 3/5 = 0.60$

### O que esperar
- **SMX** trabalha em **zonas definidas por especialistas** (os cortes espectrais vêm do JSON de configuração), então tende a ter taxa de concordância alta.
- **SHAP e Permutation** operam feature-a-feature e depois se agrega por zona — podem captar regiões do espectro ruidosas ou de background.
- **VIP** para PLS tende a ter boa concordância pois é baseado nos componentes latentes que são espectralmente suaves.

---

## 5. Baselines and Comparison Matrix (Matriz de Comparação)

### O que é
Definição de quais métodos são comparados em quais dimensões, garantindo **fairness** na comparação.

### Matriz de comparação

| Dimensão              | SMX | SHAP | VIP (PLS only) | Permutation |
|-----------------------|-----|------|----------------|-------------|
| Faithfulness          | ✓   | ✓    | ✓              | ✓           |
| Stability (split)     | ✓   | ✓    | ✓              | ✓           |
| Stability (model)     | ✓   | ✓    | N/A*           | ✓           |
| Model Dependence      | ✓   | ✓    | N/A**          | ✓           |
| Domain-Grounded       | ✓   | ✓    | ✓              | ✓           |
| Comprehensibility     | ✓   | ✓    | ✓              | ✓           |

\* PLS é determinístico → stability ao model-seed é trivial (RBO=1).  
\** VIP só existe para PLS → não se pode fazer cross-model.

### Regras obrigatórias
- **Splits idênticos**: todos os métodos usam exatamente os mesmos folds treino/teste (controlados por `split_seed`).
- **Modelo idêntico**: para uma dada comparação, o modelo treinado é o **mesmo objeto** — não se retreina.
- **Métricas idênticas**: a mesma implementação de F1, accuracy, RBO é usada para todos.

---

## 6. Comprehensibility (Compreensibilidade)

### O que é
Responde: **"Quão fácil é para um humano entender a explicação?"** Uma explicação que requer muitas zonas para "explicar" o modelo é menos compreensível do que uma que foca em poucas zonas decisivas.

### 6.1 $N_{80}$: Número de zonas para explicar 80% da relevância

#### Definição formal

Dada uma lista de zonas com scores de relevância $r_1 \geq r_2 \geq \ldots \geq r_Z$ (normalizados para somar 1):

$$\text{CumulRel}(n) = \sum_{j=1}^{n} r_j$$

$$N_{80} = \min \{n \mid \text{CumulRel}(n) \geq 0.80\}$$

#### Normalização dos scores

Para cada método, os scores brutos são normalizados:

$$r_j = \frac{\text{score}_j}{\sum_{j'=1}^{Z} \text{score}_{j'}}$$

onde $\text{score}_j$ pode ser:
- **SMX**: LRC da zona $j$
- **SHAP**: $\sum_{k \in z_j} \overline{|\phi^{(k)}|}$ (SHAP absoluto médio agregado)
- **VIP**: VIP médio da zona
- **Permutation**: importância por permutação da zona

#### Exemplo

Suponha 8 zonas:

| Zona  | SMX $r_j$ | SHAP $r_j$ |
|-------|-----------|------------|
| Ca ka | 0.35      | 0.20       |
| Fe ka | 0.25      | 0.18       |
| Ti ka | 0.15      | 0.17       |
| Si ka | 0.10      | 0.15       |
| Mg ka | 0.06      | 0.12       |
| Cu ka | 0.04      | 0.08       |
| Zn ka | 0.03      | 0.06       |
| Pb la | 0.02      | 0.04       |

**SMX**: CumulRel(1)=0.35, CumulRel(2)=0.60, CumulRel(3)=0.75, CumulRel(4)=0.85 → $N_{80}^{\text{SMX}} = 4$

**SHAP**: CumulRel(1)=0.20, CumulRel(2)=0.38, CumulRel(3)=0.55, CumulRel(4)=0.70, CumulRel(5)=0.82 → $N_{80}^{\text{SHAP}} = 5$

**Interpretação**: SMX é mais compreensível — concentra a importância em menos zonas. SHAP distribui mais uniformemente.

#### O que esperar
- $N_{80}$ menor → explicação mais **compacta** e mais fácil de interpretar.
- Métodos que produzem distribuições mais "peaked" (concentradas) tendem a ter $N_{80}$ menor.
- SMX, por construção do grafo + LRC, tende a concentrar importância em poucos nós centrais.

### 6.2 Match com assinaturas elementais esperadas

#### Definição formal

Defina $\mathcal{E}$ como o conjunto de zonas com assinatura elemental **esperada** (definidas por conhecimento de domínio — e.g., no dataset de solos, espera-se que Fe, Ca, K sejam relevantes).

Para as **top-K** zonas do ranking:

$$\text{elemental\_match}(K) = \frac{|\text{top-K} \cap \mathcal{E}|}{K}$$

#### Exemplo

$\mathcal{E} = \{\text{Ca ka}, \text{Fe ka}, \text{Ti ka}, \text{K ka}\}$ (4 zonas esperadas pelo especialista).

| Rank | SMX         | SHAP         |
|------|-------------|-------------|
| 1    | Ca ka ✓     | background ✗ |
| 2    | Fe ka ✓     | Fe ka ✓      |
| 3    | Ti ka ✓     | noise ✗      |
| 4    | Mg ka ✗     | Ca ka ✓      |
| 5    | K ka ✓      | Ti ka ✓      |

$\text{elemental\_match}_{\text{SMX}}(K=2) = 2/2 = 1.0$  
$\text{elemental\_match}_{\text{SHAP}}(K=2) = 1/2 = 0.5$

$\text{elemental\_match}_{\text{SMX}}(K=5) = 4/5 = 0.8$  
$\text{elemental\_match}_{\text{SHAP}}(K=5) = 3/5 = 0.6$

Reportar para $K = 2, 5, 10$ conforme o protocolo.

---

## 7. Protocolo Estatístico de Reportamento

Esta seção explica em detalhe cada conceito estatístico utilizado no protocolo. Nenhum conhecimento prévio de estatística é assumido.

---

### 7.1 O que é Cross-Validation (Validação Cruzada)

#### O problema que resolve

Quando treinamos um modelo de machine learning, precisamos saber **quão bem ele vai funcionar em dados que nunca viu**. Se usarmos todos os dados para treinar e depois testarmos nos mesmos dados, o resultado é otimista demais — o modelo "decorou" os dados (overfitting). Precisamos separar parte dos dados para testar.

A solução mais simples é dividir os dados em dois grupos: **treino** (para construir o modelo) e **teste** (para avaliá-lo). Mas essa divisão única é arriscada: se por acaso colocamos amostras "fáceis" no teste e "difíceis" no treino (ou vice-versa), a estimativa de performance fica distorcida.

**Cross-validation** resolve isso fazendo **múltiplas divisões** e calculando a média dos resultados.

#### K-Fold Cross-Validation (CV de K folds)

O procedimento K-fold divide os dados em $K$ pedaços (chamados **folds**) de tamanho aproximadamente igual. Em cada rodada, um fold diferente é usado como teste e os $K-1$ restantes como treino:

```
Dados totais: [A][B][C][D][E]    (K=5 folds)

Rodada 1: Treino = [B][C][D][E], Teste = [A]  →  score₁
Rodada 2: Treino = [A][C][D][E], Teste = [B]  →  score₂
Rodada 3: Treino = [A][B][D][E], Teste = [C]  →  score₃
Rodada 4: Treino = [A][B][C][E], Teste = [D]  →  score₄
Rodada 5: Treino = [A][B][C][D], Teste = [E]  →  score₅
```

Resultado final = média dos 5 scores: $\bar{s} = \frac{1}{K}\sum_{k=1}^{K} s_k$

**Vantagem**: toda amostra é usada exatamente uma vez como teste.

#### O que significa "Stratified" (Estratificado)

Em problemas de classificação, as classes podem ser desbalanceadas (e.g., 80 amostras da Classe A e 20 da Classe B). Se a divisão em folds for puramente aleatória, podemos ter por azar um fold com 0 amostras da Classe B — o teste nesse fold seria inútil.

**Stratified K-fold** garante que **cada fold preserva a proporção original das classes**:

```
Dados totais: 80 amostras Classe A, 20 amostras Classe B  (ratio 4:1)

Fold 1: 16 A + 4 B = 20 amostras (ratio 4:1 ✓)
Fold 2: 16 A + 4 B = 20 amostras (ratio 4:1 ✓)
Fold 3: 16 A + 4 B = 20 amostras (ratio 4:1 ✓)
Fold 4: 16 A + 4 B = 20 amostras (ratio 4:1 ✓)
Fold 5: 16 A + 4 B = 20 amostras (ratio 4:1 ✓)
```

Isso garante que o modelo sempre treina e testa com representação proporcional de cada classe.

#### O que significa "Repeated" (Repetido)

Uma única rodada de K-fold depende de **como os dados foram embaralhados** antes de serem particionados. Embaralhamentos diferentes produzem folds diferentes e, consequentemente, scores ligeiramente diferentes.

**Repeated K-fold** repete todo o procedimento $R$ vezes, cada vez com um embaralhamento diferente:

- Repetição 1: embaralhar dados → 5-fold CV → 5 scores
- Repetição 2: embaralhar dados (diferente) → 5-fold CV → 5 scores

Total de scores = $K \times R$. Isso reduz a variância da estimativa.

#### 5×2 Repeated Stratified K-Fold (o que o protocolo recomenda)

O protocolo recomenda: **K=5 folds, R=2 repetições, com estratificação**.

Isso produz $5 \times 2 = 10$ avaliações independentes por `split_seed`.

```
split_seed = 42:
  Repetição 1 (shuffle A):
    Fold 1: treino=[B,C,D,E], teste=[A] → F1₁
    Fold 2: treino=[A,C,D,E], teste=[B] → F1₂
    Fold 3: treino=[A,B,D,E], teste=[C] → F1₃
    Fold 4: treino=[A,B,C,E], teste=[D] → F1₄
    Fold 5: treino=[A,B,C,D], teste=[E] → F1₅
  Repetição 2 (shuffle B):
    Fold 1: treino=[B',C',D',E'], teste=[A'] → F1₆
    Fold 2: treino=[A',C',D',E'], teste=[B'] → F1₇
    ...
    Fold 5: → F1₁₀
```

Com 10 `split_seed`s diferentes, temos $10 \times 10 = 100$ avaliações totais. Essa quantidade é suficiente para calcular médias e intervalos de confiança robustos.

**Por que 5×2 e não 10×1 ou 3×3?** O esquema 5×2 é um bom equilíbrio entre:
- número suficiente de avaliações (10),
- tamanho razoável do conjunto de treino (~80% dos dados em cada fold),
- variabilidade controlada (2 embaralhamentos diferentes).

Dietterich (1998) [R14] especificamente analisou e recomendou o esquema 5×2 para comparação de algoritmos de classificação.

---

### 7.2 Como funcionam os Seeds (sementes aleatórias) neste protocolo

Computadores não geram números verdadeiramente aleatórios — usam algoritmos que produzem sequências **pseudo-aleatórias**. O **seed** é o número inicial que alimenta esse algoritmo. **Mesmo seed → mesma sequência → mesmos resultados.** Isso garante reprodutibilidade.

O protocolo separa dois seeds com funções distintas:

#### `split_seed` — controla a divisão dos dados

Determina:
- A ordem do embaralhamento (shuffle) das amostras.
- Quais amostras vão para cada fold de treino/teste.

**Não afeta**: o treinamento do modelo em si.

#### `model_seed` — controla a aleatoriedade do modelo

Determina:
- Inicialização dos pesos (MLP).
- Ordem de processamento de mini-batches (SGD).
- Desempate em algoritmos (SVM).

**Não afeta**: a divisão treino/teste.

#### Por que separar?

Separar permite isolar as fontes de variação:

| Experimento | `split_seed` | `model_seed` | O que se mede |
|-------------|--------------|--------------|---------------|
| Estabilidade ao split | **varia** | fixo | A explicação muda quando os dados mudam? |
| Estabilidade ao modelo | fixo | **varia** | A explicação muda quando o modelo é reinicializado? |
| Variabilidade total | **varia** | **varia** | Variação combinada de ambas as fontes |

Se apenas um seed fosse usado para tudo, não saberíamos se a instabilidade vem dos dados ou do modelo.

---

### 7.3 Intervalo de Confiança (IC 95%) via Bootstrap

#### O que é um Intervalo de Confiança

Um intervalo de confiança (IC) expressa a **incerteza** sobre uma estimativa. Quando dizemos "F1 = 0.85, IC 95%: [0.80, 0.90]", estamos dizendo:

> "Nosso melhor palpite para o F1 verdadeiro é 0.85, e estamos 95% confiantes de que o valor real está entre 0.80 e 0.90."

Sem IC, reportar apenas "F1 = 0.85" não diz se isso é estável (sempre dá ~0.85) ou muito variável (às vezes dá 0.60, às vezes 0.99, e na média dá 0.85).

#### O que é o Bootstrap

O bootstrap é um método de reamostragem inventado por Efron (1979) [R13] para estimar a distribuição de uma estatística **sem assumir nenhuma distribuição teórica** (como normalidade). É um dos métodos mais versáteis da estatística moderna.

**Ideia central**: "Se eu pudesse repetir o experimento milhares de vezes, veria quanta variação existe na minha estimativa. Como não posso repetir o experimento, vou *simular* repetições reamostrando os dados que já tenho."

#### Como o Bootstrap funciona — passo a passo detalhado

Suponha que temos $N = 100$ scores de F1 (um por combinação de split_seed × fold):

$$\text{scores originais} = \{s_1, s_2, s_3, \ldots, s_{100}\}$$

A média observada é $\hat{\theta} = \frac{1}{100}\sum_{i=1}^{100} s_i$, digamos $\hat{\theta} = 0.85$.

**Procedimento bootstrap** (com $B = 2000$ reamostragens):

**Passo 1**: Reamostragem $b = 1$:
- Sorteie $N = 100$ valores **com reposição** do conjunto original. Isso significa que cada sorteio é independente: pega-se um valor, anota-se, e ele **volta** para o conjunto antes do próximo sorteio.
- Resultado: uma amostra bootstrap $\{s_1^{*}, s_2^{*}, \ldots, s_{100}^{*}\}$ que pode ter repetições e omissões.
- Exemplo: se os scores são $\{0.80, 0.85, 0.90, 0.88, \ldots\}$, uma reamostragem poderia ser $\{0.85, 0.85, 0.90, 0.80, 0.90, \ldots\}$ — note que 0.85 apareceu 2 vezes e algum outro valor pode não aparecer.
- Calcule a média desta reamostragem: $\bar{\theta}_1^{*} = \frac{1}{100}\sum s_i^{*}$, digamos $\bar{\theta}_1^{*} = 0.846$.

**Passo 2**: Repita o Passo 1 para $b = 2, 3, \ldots, 2000$:

$$\bar{\theta}_1^{*} = 0.846, \quad \bar{\theta}_2^{*} = 0.853, \quad \bar{\theta}_3^{*} = 0.849, \quad \ldots, \quad \bar{\theta}_{2000}^{*} = 0.851$$

Agora temos 2000 médias bootstrap, que formam uma **distribuição empírica** da estatística.

**Passo 3**: Ordene as 2000 médias do menor para o maior:

$$\bar{\theta}_{(1)}^{*} \leq \bar{\theta}_{(2)}^{*} \leq \ldots \leq \bar{\theta}_{(2000)}^{*}$$

**Passo 4**: O IC 95% é dado pelos **percentis 2.5% e 97.5%** dessa distribuição ordenada:

$$\text{IC}_{95\%} = \left[\bar{\theta}_{(50)}^{*}, \; \bar{\theta}_{(1950)}^{*}\right]$$

(O valor na posição 50 é o percentil 2.5% de 2000 valores, e a posição 1950 é o percentil 97.5%.)

#### Exemplo numérico completo

Temos 10 scores de F1-drop: $\{0.12, 0.15, 0.08, 0.14, 0.11, 0.13, 0.09, 0.16, 0.10, 0.14\}$

Média observada: $\hat{\theta} = 0.122$

Fazemos $B = 2000$ reamostragens bootstrap (cada uma sorteia 10 valores com reposição dos 10 originais). Suponha que as 2000 médias bootstrap, depois de ordenadas, têm:
- Posição 50 (percentil 2.5%): $0.096$
- Posição 1950 (percentil 97.5%): $0.148$

**Resultado**: F1-drop = 0.122, IC 95% = [0.096, 0.148]

**Interpretação**: "Com 95% de confiança, a queda verdadeira de F1 ao remover as top zones está entre 0.096 e 0.148."

#### Por que "com reposição" é essencial

Reamostrar **com reposição** significa que em cada reamostragem bootstrap, algumas observações originais podem aparecer 0 vezes, 1 vez, 2 vezes ou mais. É isso que introduz variabilidade entre as reamostragens e permite estimar a incerteza. Se reamostrássemos **sem reposição**, cada reamostragem seria idêntica ao conjunto original (apenas com ordem diferente), e todas as médias seriam iguais — inútil.

#### Por que 2000 reamostragens

Com $B = 2000$, os percentis 2.5% e 97.5% são estimados com boa precisão. Valores menores (e.g., $B = 100$) resultam em ICs instáveis — executar o bootstrap de novo daria percentis diferentes. A partir de $B = 1000$ a estabilidade é geralmente aceitável; $B = 2000$ dá margem adicional de segurança [R13].

#### Vantagens do bootstrap sobre métodos clássicos

| Propriedade | Bootstrap | IC clássico (fórmula $\bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}$) |
|-------------|-----------|------------------------------------------------------------------|
| Assume normalidade? | **Não** | Sim |
| Funciona com distribuições assimétricas? | **Sim** | Mal |
| Funciona com amostras pequenas? | **Sim** (com cuidado) | Não confiável |
| Aplicável a qualquer estatística (mediana, RBO, etc.)? | **Sim** | Apenas para médias e proporções |

No nosso protocolo, bootstrap é essencial porque métricas como RBO e AUDC **não têm distribuição teórica conhecida**, então fórmulas clássicas de IC não se aplicam.

---

### 7.4 Teste Pareado (comparação método vs. método)

#### O problema

Queremos saber: **"O método A é realmente melhor que o método B, ou a diferença observada é apenas ruído?"**

Exemplo: SMX obteve AUDC = 0.72 e SHAP obteve AUDC = 0.68. Isso é uma diferença **real** ou poderia ser explicada por flutuação aleatória?

#### Por que teste "pareado" (e não independente)

Cada método é avaliado nos **mesmos folds/seeds**. Isso significa que as observações estão **emparelhadas** — na rodada 1, tanto SMX quanto SHAP viram o mesmo conjunto de teste. Ignorar esse emparelhamento desperdiçaria informação e reduziria o poder estatístico.

O emparelhamento é como uma corrida: em vez de comparar os tempos brutos de dois corredores em dias diferentes (muita variação por vento, temperatura), você os faz correr **na mesma pista, no mesmo dia**, lado a lado. A diferença entre eles no mesmo dia é muito mais informativa.

#### Teste t pareado — quando usar e como funciona

**Quando usar**: quando as diferenças pareadas $d_i$ têm distribuição aproximadamente simétrica e sem outliers extremos (verificável com histograma ou teste de Shapiro-Wilk).

**Procedimento passo a passo**:

1. Para cada run $i = 1, \ldots, n$ (e.g., $n = 100$ combinações seed×fold), calcular a diferença:
$$d_i = \text{métrica}_{m_1}^{(i)} - \text{métrica}_{m_2}^{(i)}$$

   Exemplo com $n = 5$ runs:
   | Run | AUDC (SMX) | AUDC (SHAP) | $d_i$ |
   |-----|------------|-------------|-------|
   | 1   | 0.75       | 0.70        | +0.05 |
   | 2   | 0.68       | 0.72        | -0.04 |
   | 3   | 0.80       | 0.65        | +0.15 |
   | 4   | 0.71       | 0.69        | +0.02 |
   | 5   | 0.76       | 0.68        | +0.08 |

2. Calcular a média e o desvio padrão das diferenças:
$$\bar{d} = \frac{1}{n}\sum_{i=1}^{n} d_i = \frac{0.05 + (-0.04) + 0.15 + 0.02 + 0.08}{5} = 0.052$$

$$s_d = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(d_i - \bar{d})^2} = \sqrt{\frac{(0.05-0.052)^2 + (-0.04-0.052)^2 + \ldots}{4}} \approx 0.071$$

3. Calcular a estatística $t$:
$$t = \frac{\bar{d}}{s_d / \sqrt{n}} = \frac{0.052}{0.071 / \sqrt{5}} = \frac{0.052}{0.0318} \approx 1.64$$

4. Comparar com a distribuição $t$ com $n-1 = 4$ graus de liberdade. O valor crítico bicaudal para $\alpha = 0.05$ é $t_{0.025, 4} = 2.776$.

   Como $1.64 < 2.776$, **não rejeitamos** a hipótese nula (a diferença pode ser ruído).

5. O p-valor é a probabilidade de observar um $t$ tão extremo ou mais se não houvesse diferença real. Se $p < 0.05$, consideramos a diferença **estatisticamente significativa**.

**Hipóteses do teste**:
- $H_0$: $\mu_d = 0$ (não há diferença real entre os métodos)
- $H_1$: $\mu_d \neq 0$ (há diferença real)

#### Teste de Wilcoxon signed-rank — quando usar e como funciona

**Quando usar**: quando as diferenças $d_i$ **não** são aproximadamente normais (distribuição assimétrica, com outliers, ou amostra muito pequena).

O Wilcoxon é a alternativa **não-paramétrica** — não assume nenhuma forma de distribuição.

**Procedimento passo a passo**:

1. Calcular as diferenças $d_i$ (igual ao teste t).

2. Remover diferenças iguais a zero ($d_i = 0$) e anotar n' restantes.

3. Calcular os **valores absolutos** $|d_i|$ e **ranqueá-los** do menor para o maior (ignorando o sinal):

   | $d_i$ | $|d_i|$ | Rank |
   |-------|---------|------|
   | +0.02 | 0.02    | 1    |
   | -0.04 | 0.04    | 2    |
   | +0.05 | 0.05    | 3    |
   | +0.08 | 0.08    | 4    |
   | +0.15 | 0.15    | 5    |

4. Separar os ranks conforme o **sinal original** de $d_i$:
   - Ranks positivos ($d_i > 0$): $R^+ = \{1, 3, 4, 5\}$, soma $W^+ = 1 + 3 + 4 + 5 = 13$
   - Ranks negativos ($d_i < 0$): $R^- = \{2\}$, soma $W^- = 2$

5. A estatística do teste é:
$$W = \min(W^+, W^-) = \min(13, 2) = 2$$

6. Comparar $W$ com o valor crítico tabelado para $n' = 5$ e $\alpha = 0.05$. Se $W \leq W_{\text{crítico}}$, rejeita-se $H_0$.

**Intuição**: se todos os $d_i$ fossem positivos (método 1 sempre melhor), $W^- = 0$. Quanto menor $W$, mais forte a evidência de diferença.

#### Como decidir entre teste t e Wilcoxon

Antes de rodar o teste, verifique a normalidade das diferenças $d_i$:

1. **Histograma visual**: se for aproximadamente simétrico e em forma de sino → teste t.
2. **Teste de Shapiro-Wilk**: $H_0$ = dados são normais. Se $p > 0.05$ → pode usar teste t. Se $p \leq 0.05$ → usar Wilcoxon.

Na dúvida, **use Wilcoxon** — é mais conservador e sempre válido.

---

### 7.5 Tamanho de efeito (Effect Size)

#### O que é e por que o p-valor não basta

O p-valor diz se a diferença é **estatisticamente significativa**, mas não diz se é **praticamente relevante**. Com $n = 100$ runs, até uma diferença minúscula (e.g., 0.001 no F1) pode ser significativa. O tamanho de efeito quantifica **quão grande é a diferença** em termos padronizados.

#### Cohen's d para diferenças pareadas

$$d_{\text{Cohen}} = \frac{\bar{d}}{s_d}$$

onde:
- $\bar{d}$ = média das diferenças pareadas
- $s_d$ = desvio padrão das diferenças pareadas

**Interpretação** (convenção de Cohen, 1988):

| $|d_{\text{Cohen}}|$ | Interpretação | Exemplo prático |
|-------|---------------|-----------------|
| < 0.2 | Negligível | Diferença imperceptível na prática |
| 0.2–0.5 | Pequeno | Diferença existe mas é sutil |
| 0.5–0.8 | Médio | Diferença claramente notável |
| > 0.8 | Grande | Diferença substancial e prática |

#### Exemplo numérico

Do exemplo anterior: $\bar{d} = 0.052$, $s_d = 0.071$

$$d_{\text{Cohen}} = \frac{0.052}{0.071} \approx 0.73 \quad (\text{efeito médio})$$

**Interpretação completa**: "Embora o teste t não tenha atingido significância $(p > 0.05)$ com apenas 5 runs, o tamanho de efeito é médio $(d = 0.73)$. Com mais runs, essa diferença provavelmente seria significativa."

Isso ilustra por que reportar p-valor E tamanho de efeito é importante: um pode compensar a limitação do outro.

---

### 7.6 Resumo: como tudo se conecta no protocolo

```
100 runs (10 split_seeds × 10 folds)
    │
    ├── Para cada métrica (F1-drop, RBO, N80, etc.):
    │   │
    │   ├── Calcular média ± IC 95% (via bootstrap com B=2000)
    │   │     → "F1-drop(k=3) = 0.12 [0.09, 0.15]"
    │   │
    │   ├── Comparar pares de métodos:
    │   │   ├── Verificar normalidade das diferenças (Shapiro-Wilk)
    │   │   ├── Se normal → teste t pareado → p-valor
    │   │   ├── Se não normal → Wilcoxon signed-rank → p-valor
    │   │   └── Cohen's d → tamanho de efeito
    │   │     → "SMX vs SHAP: p = 0.003, d = 0.85 (grande)"
    │   │
    │   └── Montar tabela-resumo para o paper
    │
    └── Exemplo de tabela final:

    | Métrica    | SMX            | SHAP           | p-valor | Cohen's d |
    |------------|----------------|----------------|---------|-----------|
    | AUDC       | 0.72 [.68,.76] | 0.65 [.60,.70] | 0.003   | 0.85      |
    | RBO_split  | 0.81 [.76,.86] | 0.55 [.48,.62] | <0.001  | 1.20      |
    | N80        | 3.2 [2.8,3.6]  | 4.8 [4.2,5.4]  | 0.01    | 0.68      |
```

Referências estatísticas: [R12], [R13], [R14].

---

## 8. Artefatos de Saída Esperados

O protocolo define CSVs específicos. Resumo do que cada um contém e para que serve:

| CSV | Colunas-chave | Serve para |
|-----|---------------|------------|
| `faithfulness_zone_curve.csv` | split_seed, model_seed, fold, model, method, k, f1_drop, acc_drop | Plotar curvas de degradação por método e calcular AUDC |
| `stability_rbo.csv` | method, stability_mode, run_i, run_j, rbo | Calcular média ± CI do RBO por método e modo |
| `model_dependence_rbo.csv` | method, model_a, model_b, rbo | Quantificar quanto cada método depende do algorimo |
| `domain_grounded_expert_review.csv` | method, model, item_id, rank, plausible_yes_no | Calcular agreement_rate por método |
| `comprehensibility_metrics.csv` | method, model, n80_zones, elemental_match_k5/k10 | Comparar compacidade e domain-match |
| `xai_metrics_summary.csv` | mean, CI, p-value por métrica e par de métodos | Tabela final do paper |

---

## Resumo Visual do Protocolo

```
Para cada split_seed × fold × model:
│
├── Treinar modelo (PLS / SVM / MLP)
│
├── Para cada método (SMX / SHAP / VIP / Perm):
│   ├── Extrair ranking de zonas
│   ├── → Faithfulness: mascarar top-k, medir F1-drop
│   ├── → Comprehensibility: calcular N80, elemental_match
│   └── → Domain-grounded: apresentar ao especialista
│
├── Stability: variar seeds, calcular RBO pareado
│
└── Model Dependence: mesmo split, comparar rankings entre modelos via RBO
```

Cada dimensão responde a uma pergunta diferente sobre a qualidade da explicação, e **juntas** dão um panorama completo: a explicação é fiel? estável? independente do modelo? compreensível? plausível no domínio? 