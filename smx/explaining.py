from typing import Dict, List, Tuple, Optional, Union, Callable, Literal
import warnings
import pandas as pd
import numpy as np

def extract_spectral_zones(Xcal, cuts):
    """
    Extract spectral zones from a DataFrame based on specified cuts.
    
    Parameters
    ----------
    - **Xcal** : pd.DataFrame
        DataFrame with spectral data, where columns are wavelengths/energies.
    - **cuts** : list of tuples/lists or dicts
        Each item defines a spectral zone to extract.
        - If tuple/list: (start, end) or (name, start, end)
        - If dict: {'name': str, 'start': float, 'end': float}
    
    Returns
    -------
    - **zones** : dict
        Dictionary where keys are zone names and values are DataFrames with the extracted spectral zones.
    """
    import numpy as np
    import pandas as pd

    # convert the column names to numeric when possible (NaN when not convertible)
    col_nums = pd.to_numeric(Xcal.columns.astype(str), errors='coerce')
    zones = {} # dictionary to store extracted zones

    for cut in cuts:
        # normalize cut format
        if isinstance(cut, dict): # if dict
            name = cut.get('name', f"{cut.get('start')}-{cut.get('end')}") # default name if not provided
            start = cut.get('start') # getting start value
            end = cut.get('end') # getting end value
        elif isinstance(cut, (list, tuple)): # if list/tuple
            if len(cut) == 2: 
                start, end = cut # getting start and end values
                name = f"{start}-{end}" # default name
            elif len(cut) == 3: # if name provided
                name, start, end = cut # getting name, start and end values
            else:
                raise ValueError("Cuts in tuple/list format must have 2 or 3 elements.")
        else:
            raise ValueError("Each cut must be a dict or a tuple/list.")

        # validate start and end
        try:
            s = float(start)
            e = float(end)
        except Exception: # Exception for conversion errors
            raise ValueError("star and end must be numeric values (int/float or convertible strings).")

        if s > e: # swap if necessary
            s, e = e, s

        # to select columns whose numeric value is in the interval [s, e]
        mask = (~np.isnan(col_nums)) & (col_nums >= s) & (col_nums <= e)
        selected_cols = list(Xcal.columns[mask])

        # piecing the zone DataFrame into the dictionary
        zones[name] = Xcal.loc[:, selected_cols]

    return zones

def aggregate_spectral_zones(spectral_zones_dict, aggregator='sum'):
    """
    Agrega os valores das zonas espectrais usando diferentes funções de agregação.
    
    Esta função processa cada zona espectral (DataFrame com múltiplas colunas de energia)
    e reduz cada linha (amostra) a um único valor numérico usando a função de agregação
    especificada.
    
    Parameters
    ----------
    - **spectral_zones_dict** : dict
        Dicionário retornado por extract_spectral_zones, onde:
        - chaves = nomes das zonas espectrais (ex: 'Ca ka', 'Fe ka')
        - valores = DataFrames com dados espectrais (linhas=amostras, colunas=energias)
    
    - **aggregator** : str, opcional (padrão='sum')
        Função de agregação a aplicar nas colunas de cada zona. Opções:
        - **'sum'**: Soma de todos os valores da zona (padrão)
        - **'mean'**: Média aritmética dos valores
        - **'median'**: Mediana dos valores
        - **'max'**: Valor máximo na zona
        - **'min'**: Valor mínimo na zona
        - **'std'**: Desvio padrão dos valores
        - **'var'**: Variância dos valores
        - **'extreme'**: Valor de maior magnitude (mais intenso) na zona, ou seja,
          escolhe o valor com maior valor absoluto em cada amostra (pode ser positivo ou negativo)
    
    Returns
    -------
    - **aggregated_df** : pd.DataFrame
        DataFrame com valores agregados, onde:
        - linhas = amostras (mesmo índice dos DataFrames originais)
        - colunas = zonas espectrais
        - valores = resultado da agregação (mesmo formato que .sum(axis=1))
    
    Raises
    ------
    - ValueError
        Se o agregador especificado não for reconhecido.
    """
    import pandas as pd
    import numpy as np
    
    # VALIDAÇÃO DE ENTRADA
    valid_aggregators = ['sum', 'mean', 'median', 'max', 'min', 'std', 'var', 'extreme']
    
    if aggregator not in valid_aggregators:
        raise ValueError(
            f"Agregador '{aggregator}' não reconhecido.\n"
            f"Opções válidas: {', '.join(valid_aggregators)}"
        )
    
    # MAPEAMENTO DOS AGREGADORES
    # Dicionário que mapeia strings para funções do pandas
    aggregation_functions = {
        'sum': lambda df: df.sum(axis=1),        # soma ao longo das colunas
        'mean': lambda df: df.mean(axis=1),      # média
        'median': lambda df: df.median(axis=1),  # mediana
        'max': lambda df: df.max(axis=1),        # valor máximo
        'min': lambda df: df.min(axis=1),        # valor mínimo
        'std': lambda df: df.std(axis=1),        # desvio padrão
        'var': lambda df: df.var(axis=1),        # variância
        # 'extreme': escolhe o valor com maior magnitude (abs), preservando o sinal
        'extreme': lambda df: df.apply(
            lambda row: (row.loc[row.abs().idxmax()] if row.notna().any() else np.nan),
            axis=1
        ),
    }
    
    # AGREGAÇÃO DAS ZONAS ESPECTRAIS
    aggregated_dict = {}  # dicionário para armazenar resultados
    
    for zone_name, zone_df in spectral_zones_dict.items():
        # Aplica a função de agregação selecionada
        # O resultado é uma Series (mesma estrutura que .sum(axis=1))
        aggregated_series = aggregation_functions[aggregator](zone_df)
        
        # Armazena no dicionário
        aggregated_dict[zone_name] = aggregated_series
    
    # CONSTRUÇÃO DO DATAFRAME FINAL
    # Cada chave vira uma coluna, preservando os índices originais
        aggregated_df = pd.DataFrame(aggregated_dict)    
    return aggregated_df

def predicates_by_quantiles(zone_sums_df, quantiles):
    """
    Generate predicates based on specified quantiles for each column in a DataFrame
    and create a predicate indicator matrix.
    
    Parameters
    ----------
    - **zone_sums_df** : pd.DataFrame
        DataFrame with summed values for spectral zones.
    - **quantiles** : list of float
        List of quantiles (between 0 and 1) to generate predicates for.
    
    Returns
    -------
    - **predicates_df** : pd.DataFrame
        DataFrame containing the generated predicates with columns:
        'predicate', 'rule', 'zone', 'thresholds', 'operator'.
    - **predicate_indicator_df** : pd.DataFrame
        Binary indicator matrix (samples × predicates) where 1 indicates
        the sample satisfies the predicate, 0 otherwise.
    """
    import pandas as pd
    import numpy as np

    # calculating the quantiles for each column of zone_sums_df
    zone_quantiles = zone_sums_df.quantile(quantiles)
    
    zone_predicate_list = []
    predicate_num = 1
    for zone in zone_sums_df.columns:
        for q in quantiles:
            q_value = zone_quantiles.loc[q, zone]
            # <= Q
            zone_predicate_list.append({
                'predicate': f'P{predicate_num}',
                'rule': f"{zone} <= {q_value:.2f}",
                'zone': zone,
                'thresholds': f"{q_value:.2f}",
                'operator': "<="
            })
            predicate_num += 1
            # > Q
            zone_predicate_list.append({
                'predicate': f'P{predicate_num}',
                'rule': f"{zone} > {q_value:.2f}",
                'zone': zone,
                'thresholds': f"{q_value:.2f}",
                'operator': ">"
            })
            predicate_num += 1
    
    predicates_df = pd.DataFrame(zone_predicate_list)
    
    # Removing duplicate predicates based on 'rule' column
    # Some zones may have the same quantile values, creating duplicate rules
    initial_count = len(predicates_df)
    predicates_df = predicates_df.drop_duplicates(subset=['rule'], keep='first').reset_index(drop=True)
    final_count = len(predicates_df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate predicates. Remaining: {final_count}")
    
    # Renumbering predicates after removing duplicates
    predicates_df['predicate'] = [f'P{i+1}' for i in range(len(predicates_df))]

    # Generating the predicate indicator DataFrame
    
    # function to evaluate a predicate for a given value
    def eval_predicate(value, thresholds, operator):
        if operator == "<=":
            return float(value <= float(thresholds))
        elif operator == ">":
            return float(value > float(thresholds))
        else:
            return np.nan
    
    # compute all columns first, then concatenate them at once
    columns_dict = {}
    
    # iterating over each predicate
    for _, row in predicates_df.iterrows():
        pred = row['predicate']
        zone = row['zone']
        thresholds = row['thresholds']
        operator = row['operator']
        columns_dict[pred] = zone_sums_df[zone].apply(
            lambda v: eval_predicate(v, thresholds, operator)
        ).astype(int)
    
    # create DataFrame from all columns at once
    predicate_indicator_df = pd.DataFrame(columns_dict, index=zone_sums_df.index)
    
    # setting column names to rules for better readability
    predicate_indicator_df.columns = predicates_df['rule'].tolist()
    
    # computing co-occurrence matrix
    co_occurrence_matrix = np.dot(predicate_indicator_df.T, predicate_indicator_df)
    co_occurrence_matrix_df = pd.DataFrame(co_occurrence_matrix, index=predicate_indicator_df.columns, columns=predicate_indicator_df.columns) 

    return predicates_df, predicate_indicator_df, co_occurrence_matrix_df

def create_predicate_info_dict(predicates_df, predicate_indicator_df, zone_aggregated_df, y_predicted_numeric):
    """
    Cria um dicionário com informações detalhadas sobre cada predicado.
    
    Para cada predicado, armazena:
    - Os valores agregados da zona espectral correspondente (das amostras que satisfazem o predicado)
    - Os valores preditos pelo modelo (das mesmas amostras)
    - Opcionalmente: índices das amostras, classe predita, etc.
    
    Parameters
    ----------
    - **predicates_df** : pd.DataFrame
        DataFrame com predicados gerados por `predicates_by_quantiles()` ou similar.
        Colunas obrigatórias: ['predicate', 'rule', 'zone', 'thresholds', 'operator']
        
    - **predicate_indicator_df** : pd.DataFrame
        Matriz binária de indicadores (samples × predicates) retornada por `predicates_by_quantiles()`.
        Colunas são as regras dos predicados (ex: "Ca ka <= 25.5")
        Valores: 1 = amostra satisfaz o predicado, 0 = não satisfaz
        
    - **zone_aggregated_df** : pd.DataFrame
        DataFrame com valores agregados das zonas espectrais (retornado por `aggregate_spectral_zones()`).
        Linhas = amostras, Colunas = zonas espectrais
        Valores = resultado da agregação (sum, mean, median, std, etc.)
        
    - **y_predicted_numeric** : pd.Series, pd.DataFrame ou np.ndarray
        Valores preditos pelo modelo (contínuos).
        - Para PLS-DA: valores entre 0 e 1 (ex: `plsda_results[5].iloc[:, -1]`)
        - Para PLS-R: valores contínuos da variável resposta
        - Deve ter o mesmo número de linhas que `zone_aggregated_df`
    
    Returns
    -------
    - **predicate_info_dict** : dict
        Dicionário estruturado como:
        {
            'Ca ka <= 25.5': DataFrame({
                'Zone_Aggregated': [valores agregados da zona Ca ka],
                'Predicted_Y': [valores preditos pelo modelo],
                'Sample_Index': [índices originais das amostras]
            }),
            'Fe ka > 10.2': DataFrame({...}),
            ...
        }
        
        - Chaves: Regras dos predicados (strings)
        - Valores: DataFrames com 3 colunas:
            - **Zone_Aggregated**: Valores agregados da zona espectral (pode ser soma, média, mediana, etc.)
            - **Predicted_Y**: Valores preditos pelo modelo para essas amostras
            - **Sample_Index**: Índices originais das amostras (para rastreabilidade)
    
    Raises
    ------
    - ValueError
        Se os DataFrames de entrada tiverem número incompatível de amostras
    - KeyError
        Se alguma coluna obrigatória estiver faltando
    """
    import pandas as pd
    import numpy as np
    
    # VALIDAÇÃO DE ENTRADAS
    
    # Verificar colunas obrigatórias em predicates_df
    required_cols = ['predicate', 'rule', 'zone', 'thresholds', 'operator']
    missing_cols = [col for col in required_cols if col not in predicates_df.columns]
    if missing_cols:
        raise KeyError(f"Colunas faltando em predicates_df: {missing_cols}")
    
    # Converter y_predicted_numeric para Series se necessário
    if isinstance(y_predicted_numeric, pd.DataFrame):
        y_predicted_numeric = y_predicted_numeric.iloc[:, -1]  # última coluna
    elif isinstance(y_predicted_numeric, np.ndarray):
        y_predicted_numeric = pd.Series(y_predicted_numeric)
    
    # Verificar compatibilidade de tamanhos
    n_samples_zones = len(zone_aggregated_df)
    n_samples_predicted = len(y_predicted_numeric)
    n_samples_indicators = len(predicate_indicator_df)
    
    if not (n_samples_zones == n_samples_predicted == n_samples_indicators):
        raise ValueError(
            f"Número incompatível de amostras:\n"
            f"  zone_aggregated_df: {n_samples_zones}\n"
            f"  y_predicted_numeric: {n_samples_predicted}\n"
            f"  predicate_indicator_df: {n_samples_indicators}\n"
            f"Todos devem ter o mesmo número de linhas."
        )
    
    # CONSTRUÇÃO DO DICIONÁRIO DE INFORMAÇÕES
    
    predicate_info_dict = {}  # dicionário para armazenar resultados
    n_predicates_processed = 0  # contador de predicados processados
    n_predicates_empty = 0  # contador de predicados sem amostras
    
    # Iterar sobre cada predicado
    for _, row in predicates_df.iterrows():
        
        pred_rule = row['rule']  # regra do predicado (ex: "Ca ka <= 25.5")
        zone_name = row['zone']  # nome da zona espectral (ex: "Ca ka")
        
        # 1. IDENTIFICAR AMOSTRAS QUE SATISFAZEM O PREDICADO
        # Usar a matriz de indicadores para filtrar amostras
        # predicate_indicator_df tem colunas com as regras dos predicados
        
        if pred_rule not in predicate_indicator_df.columns:
            # Predicado não existe na matriz de indicadores (não deveria acontecer)
            continue
        
        # Máscara booleana: True = amostra satisfaz o predicado
        mask_satisfied = predicate_indicator_df[pred_rule] == 1
        
        # Índices das amostras que satisfazem o predicado
        # Usar np.where() para compatibilidade com todos os tipos de índices
        satisfied_indices = np.where(mask_satisfied)[0].tolist()
        
        # 2. VERIFICAR SE HÁ AMOSTRAS SATISFEITAS
        if not satisfied_indices:  # lista vazia
            n_predicates_empty += 1
            continue  # pula este predicado (não adiciona ao dicionário)
        
        # 3. EXTRAIR VALORES AGREGADOS DA ZONA ESPECTRAL
        # Valores agregados (soma, média, mediana, std, etc.) da zona correspondente
        zone_aggregated_values = zone_aggregated_df.loc[satisfied_indices, zone_name]
        
        # 4. EXTRAIR VALORES PREDITOS PELO MODELO
        predicted_values = y_predicted_numeric.iloc[satisfied_indices]
        
        # 5. CRIAR DATAFRAME COM INFORMAÇÕES DO PREDICADO
        df_predicate_info = pd.DataFrame({
            'Zone_Aggregated': zone_aggregated_values.reset_index(drop=True),  # valores agregados
            'Predicted_Y': predicted_values.reset_index(drop=True),  # valores preditos
            'Sample_Index': satisfied_indices  # índices originais (para rastreabilidade)
        })
        
        # 6. ARMAZENAR NO DICIONÁRIO
        predicate_info_dict[pred_rule] = df_predicate_info
        n_predicates_processed += 1
    
    return predicate_info_dict

def bagging_predicates(zone_sums_df, y_predicted_numeric, predicates_df, 
                          n_bags=50, n_predicates_per_bag=20, n_samples_per_bag=80, 
                          min_samples_per_predicate=5, replace=True, random_seed=42,
                          sample_bagging=True, predicate_bagging=True):
    """
    Realiza bagging de predicados com controle granular sobre amostragem.
    
    Estratégia de Bagging (Configurável):
    =====================================
    1. **Amostragem de Linhas (Amostras):**
       - sample_bagging=True: Sorteia N amostras para cada bag
       - sample_bagging=False: Usa TODAS as amostras em todos os bags
    
    2. **Amostragem de Colunas (Predicados):**
       - predicate_bagging=True: Sorteia M predicados para cada bag
       - predicate_bagging=False: Usa TODOS os predicados em todos os bags
    
    3. **Filtragem e Validação:**
       - Para cada predicado selecionado, filtra as amostras que o satisfazem
       - Descarta predicados com insuficiente cobertura (se sample_bagging=True)
    
    Parâmetros
    ----------
    zone_sums_df : pd.DataFrame
        DataFrame com somas das zonas espectrais (linhas=amostras, colunas=zonas).
        
    y_predicted_numeric : pd.Series ou np.ndarray
        Valores preditos pelo modelo (contínuos, entre 0 e 1 para PLS-DA).
        
    predicates_df : pd.DataFrame
        DataFrame com predicados. Colunas obrigatórias:
        - 'rule': Regra legível (ex: "Ca ka <= 25.5")
        - 'zone': Nome da zona espectral
        - 'thresholds': Valor do threshold
        - 'operator': "<=" ou ">"
        
    n_bags : int, default=50
        Número de bags (iterações) a criar.
        
    n_predicates_per_bag : int, default=20
        Número de predicados a sortear por bag.
        **Ignorado se predicate_bagging=False.**
        
    n_samples_per_bag : int, default=80
        Número de amostras a sortear por bag.
        **Ignorado se sample_bagging=False.**
        
    min_samples_per_predicate : int, default=5
        Mínimo de amostras que devem satisfazer um predicado para ele ser válido.
        **Aplicado apenas se sample_bagging=True.**
        
    replace : bool, default=True
        - True: Bootstrap (amostragem com reposição)
        - False: Subsampling (sem reposição)
        **Aplicado apenas se sample_bagging=True.**
        
    random_seed : int, default=42
        Semente aleatória para reprodutibilidade.
        
    sample_bagging : bool, default=True
        - True: Faz subamostragem das LINHAS (amostras variam entre bags)
        - False: Usa todas as amostras em todos os bags
        
    predicate_bagging : bool, default=True
        - True: Faz subamostragem das COLUNAS (predicados variam entre bags)
        - False: Usa todos os predicados em todos os bags
    
    Returns
    -------
    bags_dict : dict
        Dicionário estruturado como:
        {
            'Bag_1': {
                'Ca ka <= 25.5': DataFrame(['Zone_Sum', 'Predicted_Y', 'Sample_Index']),
                'Fe ka > 10.2': DataFrame([...]),
                ...
            },
            'Bag_2': {...},
            ...
        }
    
    """
    import numpy as np
    import pandas as pd
    
    # INICIALIZAÇÃO
    np.random.seed(random_seed)
    
    n_total_samples = len(zone_sums_df)
    predicate_rules = predicates_df['rule'].tolist()
    bags_dict = {}
    
    # LOOP PRINCIPAL: CRIAÇÃO DOS BAGS
    for bag_num in range(1, n_bags + 1):
        
        # 1. SELEÇÃO DE AMOSTRAS (LINHAS) - Controle via `sample_bagging`
        if sample_bagging:
            # Sorteia N amostras (bootstrap ou subsampling)
            bag_sample_indices = np.random.choice(
                range(n_total_samples),
                size=n_samples_per_bag,
                replace=replace  # True=bootstrap, False=subsampling
            )
        else:
            # Usa TODAS as amostras disponíveis
            bag_sample_indices = np.arange(n_total_samples)
        
        # 2. SELEÇÃO DE PREDICADOS (COLUNAS) - Controle via `predicate_bagging`
        if predicate_bagging:
            # Sorteia M predicados aleatoriamente (sem reposição)
            selected_predicate_rules = np.random.choice(
                predicate_rules,
                size=min(n_predicates_per_bag, len(predicate_rules)),
                replace=False
            )
        else:
            # Usa TODOS os predicados disponíveis
            selected_predicate_rules = predicate_rules
        
        # 3. FILTRAGEM E VALIDAÇÃO DE PREDICADOS
        bag_predicate_dict = {}
        n_discarded = 0
        
        for pred_rule in selected_predicate_rules:
            
            # Recupera metadados do predicado
            pred_row_filtered = predicates_df[predicates_df['rule'] == pred_rule]
            if len(pred_row_filtered) == 0:
                continue  # Predicado não encontrado, pula
            pred_row = pred_row_filtered.iloc[0]
            zone = pred_row['zone']
            threshold = float(pred_row['thresholds'])
            operator = pred_row['operator']
            
            # Extrai valores da zona para as amostras do bag
            zone_values_bag = zone_sums_df.loc[bag_sample_indices, zone].values
            
            # Aplica a regra do predicado
            if operator == "<=":
                mask_satisfied = zone_values_bag <= threshold
            elif operator == ">":
                mask_satisfied = zone_values_bag > threshold
            else:
                continue  # Operador inválido, pula
            
            # Filtra amostras que satisfazem o predicado
            satisfied_indices_in_bag = bag_sample_indices[mask_satisfied]
            
            # Validação de cobertura mínima (apenas se sample_bagging=True)
            if sample_bagging and len(satisfied_indices_in_bag) < min_samples_per_predicate:
                n_discarded += 1
                continue
            
            # Validação básica (descarta predicados vazios sempre)
            if len(satisfied_indices_in_bag) == 0:
                n_discarded += 1
                continue
            
            # Armazena dados do predicado válido
            df_predicate_info = pd.DataFrame({
                'Zone_Sum': zone_sums_df.loc[satisfied_indices_in_bag, zone].values,
                'Predicted_Y': y_predicted_numeric.iloc[satisfied_indices_in_bag].values,
                'Sample_Index': satisfied_indices_in_bag
            })
            
            bag_predicate_dict[pred_rule] = df_predicate_info
        
        # 4. ARMAZENAMENTO DO BAG
        if len(bag_predicate_dict) > 0:
            bags_dict[f'Bag_{bag_num}'] = bag_predicate_dict
            
            # Log informativo
            samp_str = "Sim" if sample_bagging else "Não"
            pred_str = f"Sim ({n_predicates_per_bag})" if predicate_bagging else "Não (Todos)"
            print(f"Bag_{bag_num} | Amostras: {samp_str} | Predicados: {pred_str} | "
                  f"Válidos: {len(bag_predicate_dict)} | Descartados: {n_discarded}")
        else:
            print(f"Bag_{bag_num}: VAZIO (todos os predicados descartados)")
    
    return bags_dict

def calculate_predicate_metrics(bags_result, metric='mutual_info', threshold=0.1, n_neighbors=10):
    """
    Calcula métricas de associação entre valores agregados das zonas espectrais 
    e as predições do modelo para cada predicado em cada bag.
    
    Esta função processa todos os bags gerados por `bagging_predicates()` e calcula
    a força da associação entre os valores das zonas espectrais e as predições contínuas
    do modelo. Suporta duas métricas: Mutual Information e Covariância.
    
    Parameters
    ----------
    - **bags_result** : dict
        Dicionário retornado por `bagging_predicates_v3()`, estruturado como:
        {
            'Bag_1': {
                'Ca ka <= 25.5': DataFrame(['Zone_Sum', 'Predicted_Y', 'Sample_Index']),
                'Fe ka > 10.2': DataFrame([...]),
                ...
            },
            'Bag_2': {...},
            ...
        }
        
    - **metric** : str, opcional (padrão='mutual_info')
        Métrica de associação a calcular. Opções:
        - **'mutual_info'**: Informação Mútua (MI) - Mede dependência não-linear
        - **'covariance'**: Covariância - Mede dependência linear
        
    - **threshold** : float, opcional (padrão=0.1)
        Valor mínimo da métrica para um predicado ser considerado relevante.
        Predicados com métrica < threshold são FILTRADOS do resultado.
        - Para MI: valores típicos entre 0.0 e 1.0+ (quanto maior, mais informativo)
        - Para Covariance: valores dependem da escala dos dados (use valores absolutos)
        
    - **n_neighbors** : int, opcional (padrão=10)
        Número de vizinhos para o cálculo de Mutual Information.
        **Usado apenas quando metric='mutual_info'. Ignorado para covariância.**
        - Valores baixos (3-5): mais sensível a ruído local
        - Valores médios (10-20): balanço entre sensibilidade e robustez (recomendado)
        - Valores altos (>30): mais suave, menos sensível a variações locais
    
    Returns
    -------
    - **metrics_results_dict** : dict
        Dicionário estruturado como:
        {
            'Bag_1': DataFrame({
                'Predicate': ['Ca ka <= 25.5', 'Fe ka > 10.2', ...],
                'Mutual_Info': [0.45, 0.32, ...]  # ou 'Covariance' se metric='covariance'
            }),
            'Bag_2': DataFrame({...}),
            ...
        }
        
        Cada DataFrame contém:
        - **Predicate**: Regra do predicado (string)
        - **Mutual_Info** ou **Covariance**: Valor da métrica calculada
        - Ordenado de forma DECRESCENTE pela métrica (maiores valores primeiro)
        - Filtrado para manter apenas predicados com métrica > threshold
    
    Raises
    ------
    - ValueError
        Se metric não for 'mutual_info' ou 'covariance'
        
    - KeyError
        Se algum bag não contiver as colunas esperadas ('Zone_Sum', 'Predicted_Y')
    
    Notes
    -----
    - **Mutual Information (MI):**
        - Captura dependências LINEARES e NÃO-LINEARES entre X e Y
        - Valores sempre >= 0 (0 = independência, >0 = dependência)
        - Mais robusto a outliers que covariância
        - Computacionalmente mais custoso
        - Ideal para relações complexas/não-lineares
    
    - **Covariância:**
        - Captura apenas dependências LINEARES
        - Valores podem ser positivos ou negativos (usamos |valor absoluto|)
        - Sensível a outliers e escala dos dados
        - Computacionalmente mais rápido
        - Ideal para relações lineares simples
    
    - **Threshold:**
        - Define o "corte de relevância" para filtrar predicados fracos
        - Valores muito baixos: mantém muitos predicados (alguns irrelevantes)
        - Valores muito altos: pode descartar predicados úteis
        - Recomendação: começar com 0.1 para MI, ajustar conforme necessidade
    """
    import pandas as pd
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression
    
    # VALIDAÇÃO DE ENTRADAS    
    valid_metrics = ['mutual_info', 'covariance']
    if metric not in valid_metrics:
        raise ValueError(
            f"Métrica '{metric}' não reconhecida.\n"
            f"Opções válidas: {', '.join(valid_metrics)}"
        )
    
    if not isinstance(bags_result, dict):
        raise TypeError("bags_result deve ser um dicionário retornado por bagging_predicates_v3()")
    
    # INICIALIZAÇÃO    
    metrics_results_dict = {}  # dicionário para armazenar resultados
    metric_name = 'Mutual_Info' if metric == 'mutual_info' else 'Covariance'
    
    total_bags = len(bags_result)
    total_predicates_processed = 0
    total_predicates_filtered = 0
    
    print(f"Calculando {metric_name} para Predicados")
    print(f"Métrica: {metric}")
    print(f"Threshold: {threshold}")
    
    # LOOP PRINCIPAL: PROCESSAR CADA BAG    
    for bag_name, predicates_dict in bags_result.items():
        
        if len(predicates_dict) == 0:
            print(f"{bag_name}: VAZIO (pulando)")
            continue
        
        # 1. CALCULAR MÉTRICA PARA CADA PREDICADO NO BAG        
        metrics = {}  # dicionário temporário {predicate_rule: metric_value}
        
        for pred_rule, df_info in predicates_dict.items():
            
            # Validar colunas necessárias
            required_cols = ['Zone_Sum', 'Predicted_Y']
            missing_cols = [col for col in required_cols if col not in df_info.columns]
            if missing_cols:
                raise KeyError(
                    f"Bag '{bag_name}', Predicado '{pred_rule}': "
                    f"Colunas faltando: {missing_cols}"
                )
            
            # Extrair dados
            X_zone = df_info['Zone_Sum'].values.reshape(-1, 1)  # valores da zona (2D para sklearn)
            y_pred = df_info['Predicted_Y'].values  # valores preditos (1D)
            
            # Verificar se há dados suficientes
            if len(X_zone) < 2:
                metrics[pred_rule] = 0.0  # não há dados suficientes para calcular métrica
                continue
            
            # Calcular métrica selecionada
            if metric == 'mutual_info':
                # Mutual Information (não-linear)
                mi_score = mutual_info_regression(
                    X_zone, 
                    y_pred, 
                    discrete_features=False,  # X é contínua
                    n_neighbors=n_neighbors,
                    random_state=42  # reprodutibilidade
                )
                metrics[pred_rule] = mi_score[0]  # MI retorna array de 1 elemento
                
            elif metric == 'covariance':
                # Covariância (linear) - usamos valor absoluto
                # np.cov retorna matriz 2x2: [[var(X), cov(X,Y)], [cov(Y,X), var(Y)]]
                # Queremos cov(X,Y) = elemento [0,1] ou [1,0]
                cov_matrix = np.cov(X_zone.flatten(), y_pred)
                covariance = cov_matrix[0, 1]  # covariância X-Y
                metrics[pred_rule] = np.abs(covariance)  # valor absoluto
        
        total_predicates_processed += len(metrics)
        
        # 2. CONVERTER PARA DATAFRAME E ORDENAR        
        metrics_df = pd.DataFrame.from_dict(
            metrics, 
            orient='index',  # chaves = índices, valores = coluna
            columns=[metric_name]
        )
        
        # Adicionar coluna de predicado
        metrics_df.insert(0, 'Predicate', metrics_df.index)
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Ordenar de forma DECRESCENTE (maiores valores = mais informativos)
        metrics_df = metrics_df.sort_values(by=metric_name, ascending=False)
        metrics_df = metrics_df.reset_index(drop=True)
        
        # 3. FILTRAR POR THRESHOLD        
        n_before_filter = len(metrics_df)
        metrics_df = metrics_df[metrics_df[metric_name] > threshold].reset_index(drop=True)
        n_after_filter = len(metrics_df)
        n_filtered = n_before_filter - n_after_filter
        
        total_predicates_filtered += n_filtered
        
        # 4. ARMAZENAR RESULTADO        
        metrics_results_dict[bag_name] = metrics_df
    
    return metrics_results_dict

def calculate_lrc(graphs_by_seed, predicates_df):
    """
    Calcula Local Reaching Centrality (LRC) para todos os nós dos grafos.
    
    A LRC mede a importância de cada nó baseada em sua capacidade de alcançar
    outros nós no grafo, ponderada pelos pesos das arestas. Nós com maior LRC
    são mais centrais/importantes na estrutura do grafo.
    
    Parameters
    ----------
    - **graphs_by_seed** : dict
        Dicionário com grafos NetworkX por semente (retornado por build_predicate_graphs).
        Estrutura: {seed1: nx.DiGraph(), seed2: nx.DiGraph(), ...}
        
    - **predicates_df** : pd.DataFrame
        DataFrame com informações dos predicados. Colunas obrigatórias:
        - 'rule': Regra do predicado (ex: "Ca ka <= 25.5")
        - 'zone': Nome da zona espectral
        - 'thresholds': Valor do threshold
        - 'operator': "<=" ou ">"
    
    Returns
    -------
    - **lrc_by_seed** : dict
        Dicionário com DataFrames de LRC para cada semente:
        {
            seed1: DataFrame(['Node', 'Local_Reaching_Centrality', 'Zone', 'Threshold', 'Operator', 'Seed']),
            seed2: DataFrame([...]),
            ...
        }
        
        Cada DataFrame contém:
        - **Node**: Nome do nó (regra do predicado ou 'Class_eut'/'Class_dist')
        - **Local_Reaching_Centrality**: Valor da LRC (quanto maior, mais importante)
        - **Zone**: Nome da zona espectral (None para nós terminais)
        - **Threshold**: Valor do threshold (None para nós terminais)
        - **Operator**: Operador da regra (None para nós terminais)
        - **Seed**: Semente aleatória usada
        
        **Ordenação**: Decrescente por LRC (nós mais importantes primeiro)
    """
    import networkx as nx
    import pandas as pd
    import numpy as np
        
    # CÁLCULO DA LRC
    
    lrc_by_seed = {}
    
    for seed, DG in graphs_by_seed.items():
        print(f"\nProcessando LRC - Semente: {seed}")
        
        # 1. CALCULAR LRC PARA CADA NÓ
        local_reaching_centrality = {
            node: nx.local_reaching_centrality(DG, node, weight='weight')
            for node in DG.nodes()
        }
        
        # Ordenar por LRC (decrescente)
        sorted_lrc = sorted(
            local_reaching_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 2. CRIAR DATAFRAME COM LRC
        lrc_df_seed = pd.DataFrame(sorted_lrc, columns=['Node', 'Local_Reaching_Centrality'])
        
        # 3. EXTRAIR METADADOS DOS PREDICADOS
        zones = []
        thresholds = []
        operators = []
        
        for node in lrc_df_seed['Node']:
            if node.startswith('Class_'):
                # Nó terminal
                zones.append(None)
                thresholds.append(None)
                operators.append(None)
            else:
                # Predicado: buscar metadados em predicates_df
                pred_row_filtered = predicates_df[predicates_df['rule'] == node]
                
                if len(pred_row_filtered) == 0:
                    # Predicado não encontrado (não deveria acontecer)
                    zones.append('Unknown')
                    thresholds.append(None)
                    operators.append(None)
                else:
                    pred_row = pred_row_filtered.iloc[0]
                    zones.append(pred_row['zone'])
                    thresholds.append(pred_row['thresholds'])
                    operators.append(pred_row['operator'])
        
        # Adicionar colunas ao DataFrame
        lrc_df_seed['Zone'] = zones
        lrc_df_seed['Threshold'] = thresholds
        lrc_df_seed['Operator'] = operators
        lrc_df_seed['Seed'] = seed
        
        # Armazenar resultado
        lrc_by_seed[seed] = lrc_df_seed
    
        return lrc_by_seed

def calculate_lrc_single_graph(graph, predicates_df):
    """
    Calcula Local Reaching Centrality (LRC) para todos os nós de um único grafo.
    
    A LRC mede a importância de cada nó baseada em sua capacidade de alcançar
    outros nós no grafo, ponderada pelos pesos das arestas. Nós com maior LRC
    são mais centrais/importantes na estrutura do grafo.
    
    Parameters
    ----------
    - **graph** : nx.DiGraph
        Grafo direcionado do NetworkX (retornado por build_fold_predicate_graph ou similar).
        
    - **predicates_df** : pd.DataFrame
        DataFrame com informações dos predicados. Colunas obrigatórias:
        - 'rule': Regra do predicado (ex: "Ca ka <= 25.5")
        - 'zone': Nome da zona espectral
        - 'thresholds': Valor do threshold
        - 'operator': "<=" ou ">"
    
    Returns
    -------
    - **lrc_df** : pd.DataFrame
        DataFrame com as seguintes colunas:
        - **Node**: Nome do nó (regra do predicado ou 'Class_A'/'Class_B')
        - **Local_Reaching_Centrality**: Valor da LRC (quanto maior, mais importante)
        - **Zone**: Nome da zona espectral (None para nós terminais)
        - **Threshold**: Valor do threshold (None para nós terminais)
        - **Operator**: Operador da regra (None para nós terminais)
        
        **Ordenação**: Decrescente por LRC (nós mais importantes primeiro)
    """
    import networkx as nx
    import pandas as pd
    import numpy as np
    
    print(f"\nProcessando LRC do grafo...")
    
    # 1. CALCULAR LRC PARA CADA NÓ
    local_reaching_centrality = {}
    for node in graph.nodes():
        try:
            lrc_val = nx.local_reaching_centrality(graph, node, weight='weight')
        except ZeroDivisionError:
            # Ocorre quando o cálculo interno do NetworkX tenta dividir por zero.
            # Neste caso, fixamos a LRC como 0.0 para manter a execução e a consistência.
            lrc_val = 0.0
        local_reaching_centrality[node] = lrc_val
    
    # Ordenar por LRC (decrescente)
    sorted_lrc = sorted(
        local_reaching_centrality.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 2. CRIAR DATAFRAME COM LRC
    lrc_df = pd.DataFrame(sorted_lrc, columns=['Node', 'Local_Reaching_Centrality'])
    
    # 3. EXTRAIR METADADOS DOS PREDICADOS
    zones = []
    thresholds = []
    operators = []
    
    for node in lrc_df['Node']:
        if node.startswith('Class_'):
            # Nó terminal
            zones.append(None)
            thresholds.append(None)
            operators.append(None)
        else:
            # Predicado: buscar metadados em predicates_df
            pred_row_filtered = predicates_df[predicates_df['rule'] == node]
            
            if len(pred_row_filtered) == 0:
                # Predicado não encontrado (não deveria acontecer)
                zones.append('Unknown')
                thresholds.append(None)
                operators.append(None)
            else:
                pred_row = pred_row_filtered.iloc[0]
                zones.append(pred_row['zone'])
                thresholds.append(pred_row['thresholds'])
                operators.append(pred_row['operator'])
    
    # Adicionar colunas ao DataFrame
    lrc_df['Zone'] = zones
    lrc_df['Threshold'] = thresholds
    lrc_df['Operator'] = operators
    
    return lrc_df    

def get_zone_columns_from_predicate(
    predicate_rule: str,
    predicates_df: pd.DataFrame,
    spectral_cuts: List[Tuple[str, float, float]],
    Xcal_columns: pd.Index
) -> List[str]:
    """
    Obtém as colunas espectrais correspondentes à zona de um predicado.
    
    Esta função identifica qual zona espectral está associada a um predicado
    e retorna a lista de colunas (variaveis) que compõem essa zona.
    
    Parameters
    ----------
    predicate_rule : str
        Regra do predicado (ex: 'F1 <= 10.5')
    predicates_df : pd.DataFrame
        DataFrame com informações dos predicados (colunas: 'rule', 'zone', etc.)
    spectral_cuts : list of tuples
        Lista de cortes espectrais no formato [(nome, inicio, fim), ...]
    Xcal_columns : pd.Index
        Índice das colunas do DataFrame de calibração (energias)
    
    Returns
    -------
    list
        Lista de nomes de colunas (strings) que compõem a zona espectral
    
    Raises
    ------
    ValueError
        Se a zona não for encontrada nos spectral_cuts
    KeyError
        Se o predicado não existir em predicates_df
    
    Example
    -------
    >>> zone_cols = get_zone_columns_from_predicate('F1 <= 10.5', predicates_df, spectral_cuts, Xcal.columns)
    >>> print(f"Zona contém {len(zone_cols)} colunas: {zone_cols[:3]}...")
    """
    # 1. Encontrar a zona associada ao predicado
    mask = predicates_df['rule'] == predicate_rule
    if not mask.any():
        raise KeyError(f"Predicado '{predicate_rule}' não encontrado em predicates_df")
    
    zone_name = predicates_df.loc[mask, 'zone'].values[0]
    
    # 2. Encontrar os limites da zona nos spectral_cuts
    zone_start, zone_end = None, None
    for cut in spectral_cuts:
        if len(cut) == 3:
            name, start, end = cut
        elif len(cut) == 2:
            start, end = cut
            name = f"{start}-{end}"
        else:
            continue
        
        if name == zone_name:
            zone_start, zone_end = float(start), float(end)
            break
    
    if zone_start is None or zone_end is None:
        raise ValueError(f"Zona '{zone_name}' não encontrada em spectral_cuts")
    
    # 3. Selecionar colunas dentro do intervalo
    # Converter nomes de colunas para numérico quando possível
    col_numeric = pd.to_numeric(Xcal_columns.astype(str), errors='coerce')
    
    # Máscara para colunas dentro do intervalo [zone_start, zone_end]
    mask_cols = (~np.isnan(col_numeric)) & (col_numeric >= zone_start) & (col_numeric <= zone_end)
    
    zone_columns = list(Xcal_columns[mask_cols])
    
    return zone_columns

def spectral_perturbation_importance(model, X, y_pred_original, spectral_cuts, 
                                      perturbation_value=0, metric='mean_abs_diff'):
    """
    Perturba regiões espectrais e avalia o impacto nas predições do modelo.
    
    Parâmetros:
    -----------
    model : estimator
        Modelo treinado (ex: PLS-DA)
    X : pd.DataFrame
        Dados espectrais originais (amostras x wavelengths)
    y_pred_original : array-like
        Predições originais do modelo
    spectral_cuts : list of tuples
        Lista de tuplas (nome_zona, inicio, fim) definindo regiões espectrais
    perturbation_value : float, default=0
        Valor a ser usado na perturbação (0 para zerar, 1 para mudar para 1, etc)
    metric : str, default='mean_abs_diff'
        Métrica para calcular a importância: 'mean_abs_diff', 'mean_diff', 'mean_relative_dev'.
        - 'mean_abs_diff': média da diferença absoluta
        - 'mean_diff': média da diferença (com sinal)
        - 'mean_relative_dev': média do desvio relativo (cuidado com divisão por zero)
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame com zona espectral e importância (diferença média nas predições)
    """
    import pandas as pd
    import numpy as np
    
    results = []
    
    for zone_name, start, end in spectral_cuts:
        # Criar cópia dos dados para perturbação
        X_perturbed = X.copy()
        # Identificar colunas dentro do range da zona espectral
        cols_to_perturb = [col for col in X.columns if start <= float(col) <= end]
        # Perturbar as colunas (mudar para o valor especificado)
        X_perturbed[cols_to_perturb] = perturbation_value
        # Fazer predição com dados perturbados
        y_pred_perturbed = model.predict(X_perturbed)
        # Calcular diferença entre predições
        if metric == 'mean_abs_diff':
            importance = np.mean(np.abs(y_pred_original - y_pred_perturbed))
        elif metric == 'mean_diff':
            importance = np.mean(y_pred_original - y_pred_perturbed)
        elif metric == 'mean_relative_dev':
            y_pred_original_safe = np.where(y_pred_original == 0, np.nan, y_pred_original)
            rel_dev = (y_pred_perturbed - y_pred_original) / y_pred_original_safe
            importance = np.nanmean(rel_dev)
        else:
            raise ValueError(f"Métrica '{metric}' não suportada. Use 'mean_abs_diff', 'mean_diff' ou 'mean_relative_dev'.")
        
        if metric == 'mean_relative_dev' or metric == 'mean_diff':
            pass  # importance já tem o sinal
            # Armazenar resultados
            results.append({
                'Zone': zone_name,
                'Start': start,
                'End': end,
                'Importance': importance,
                'Abs_Importance': np.abs(importance),
                'N_Features': len(cols_to_perturb)
            })
        else:
            # Armazenar resultados
            results.append({
                'Zone': zone_name,
                'Start': start,
                'End': end,
                'Importance': importance,
                'N_Features': len(cols_to_perturb)
            })

    # Criar DataFrame e ordenar por importância
    results_df = pd.DataFrame(results)
    if metric == 'mean_relative_dev' or metric == 'mean_diff':
        results_df = results_df.sort_values(by='Abs_Importance', ascending=False).reset_index(drop=True)
    else:
        results_df = results_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return results_df

def calculate_predicate_perturbation(
    estimator,
    Xcalclass_prep: pd.DataFrame,
    folds_struct: Dict,
    predicates_df: pd.DataFrame,
    spectral_cuts: List[Tuple[str, float, float]],
    y_calclass: Union[pd.Series, np.ndarray] = None,
    aim: str = 'regression',
    perturbation_value: float = 0,
    perturbation_mode: str = 'constant',
    stats_source: str = 'full',
    metric: str = 'mean_abs_diff',
    verbose: bool = False,
    save_detailed_results: bool = True
) -> Dict:
    """
    Calcula a importância de cada predicado usando Perturbação Espectral.
    
    Esta função é uma alternativa à permutação. Em vez de permutar valores,
    ela substitui os valores da zona espectral por um valor fixo (ex: 0), ou de acordo 
    com uma estatística como a média, mediana, máximo ou mínimo da coluna.
    Em seguida ela mede o impacto (mudança) na predição do modelo.
    
    Suporta tanto tarefas de REGRESSÃO quanto de CLASSIFICAÇÃO através do parâmetro `aim`.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Modelo treinado com método predict().
        Para classificação com certas métricas, também pode requerer:
        - predict_proba(): para métrica 'probability_shift'
        - decision_function(): para métrica 'decision_function_shift'
        
    Xcalclass_prep : pd.DataFrame
        Dataset de calibração pré-processado (n_samples × n_features)
        
    folds_struct : dict
        Estrutura de folds no formato:
        {'Fold_1': {'rule1': DataFrame, 'rule2': DataFrame, ...}, ...}
        
    predicates_df : pd.DataFrame
        DataFrame com informações dos predicados (colunas: 'rule', 'zone', etc.)
        
    spectral_cuts : list of tuples
        Lista de cortes espectrais: [(nome, inicio, fim), ...]
        
    y_calclass : pd.Series ou np.ndarray, optional
        Rótulos verdadeiros das amostras. Obrigatório para métricas de classificação
        que comparam com ground truth (ex: 'accuracy_drop', 'f1_drop').
        
    aim : str, default='regression'
        Tipo de tarefa:
        - 'regression': usa predict() e métricas numéricas contínuas
        - 'classification': usa predict(), predict_proba() ou decision_function()
                           dependendo da métrica escolhida
        
    perturbation_value : float, default=0
        Valor usado para perturbar a zona quando perturbation_mode='constant'
        
    perturbation_mode : str, default='constant'
        Modo de perturbação:
        - 'constant': usa perturbation_value para todas as colunas (comportamento original)
        - 'mean': substitui cada coluna pela sua média
        - 'median': substitui cada coluna pela sua mediana
        - 'min': substitui cada coluna pelo seu valor mínimo
        - 'max': substitui cada coluna pelo seu valor máximo
        
    stats_source : str, default='full'
        Fonte dos dados para calcular estatísticas:
        - 'full': usa todo o dataset (Xcalclass_prep)
        - 'predicate': usa apenas as amostras do predicado
        
    metric : str, default='mean_abs_diff'
        Métrica para calcular importância. As métricas disponíveis dependem do `aim`:
        
        **PARA aim='regression':**
        - 'mean_abs_diff': Média da diferença absoluta entre predições (|y_orig - y_pert|)
        - 'mean_diff': Média da diferença com sinal (y_orig - y_pert)
        - 'mean_relative_dev': Média do desvio relativo ((y_pert - y_orig) / y_orig)
        
        **PARA aim='classification':**
        - 'prediction_change_rate': Proporção de amostras que mudaram de classe após 
          perturbação. Valores de 0 a 1, onde 1 = todas mudaram. Não requer y_calclass.
          Usa: estimator.predict()
          
        - 'accuracy_drop': Queda na acurácia após perturbação (acc_orig - acc_pert).
          Valores positivos indicam queda de performance. Requer y_calclass.
          Usa: estimator.predict()
          
        - 'f1_drop': Queda no F1-score após perturbação (f1_orig - f1_pert).
          Valores positivos indicam queda de performance. Requer y_calclass.
          Usa: estimator.predict()
          
        - 'probability_shift': Média da diferença absoluta nas probabilidades preditas.
          Mede quanto as probabilidades mudam após perturbação. Não requer y_calclass.
          Usa: estimator.predict_proba() - REQUER modelo com predict_proba (ex: SVC com probability=True)
          
        - 'decision_function_shift': Média da diferença absoluta nos valores da 
          decision function. Útil para SVM e modelos lineares. Não requer y_calclass.
          Usa: estimator.decision_function() - REQUER modelo com decision_function (ex: SVC, LinearSVC)
        
    verbose : bool, default=False
        Se True, imprime detalhes do progresso
        
    save_detailed_results : bool, default=True
        Se True, salva resultados detalhados
    
    Returns
    -------
    dict
        Dicionário no formato compatível com calculate_predicate_metrics_permutation:
        {'Fold_1': DataFrame({'Predicate': [...], 'Perturbation': [...]}), ...}
        
    Notes
    -----
    Compatibilidade de métricas com modelos sklearn:
    
    | Modelo           | predict_change_rate | accuracy_drop | probability_shift | decision_function_shift |
    |------------------|---------------------|---------------|-------------------|-------------------------|
    | SVC              | ✓                   | ✓             | ✓ (probability=True) | ✓                    |
    | LinearSVC        | ✓                   | ✓             | ✗                 | ✓                       |
    | RandomForest     | ✓                   | ✓             | ✓                 | ✗                       |
    | LogisticRegression| ✓                  | ✓             | ✓                 | ✓                       |
    | KNeighbors       | ✓                   | ✓             | ✓                 | ✗                       |
    | PLSRegression*   | ✓                   | ✓             | ✗                 | ✗                       |
    
    *PLSRegression para classificação usa threshold em predict() contínuo.
    
    Examples
    --------
    >>> # Exemplo com REGRESSÃO (PLSRegression)
    >>> results = calculate_predicate_perturbation(
    ...     estimator=pls_model,
    ...     Xcalclass_prep=X_prep,
    ...     folds_struct=folds,
    ...     predicates_df=predicates,
    ...     spectral_cuts=cuts,
    ...     aim='regression',
    ...     metric='mean_abs_diff'
    ... )
    
    >>> # Exemplo com CLASSIFICAÇÃO (SVC)
    >>> results = calculate_predicate_perturbation(
    ...     estimator=svc_model,
    ...     Xcalclass_prep=X_prep,
    ...     folds_struct=folds,
    ...     predicates_df=predicates,
    ...     spectral_cuts=cuts,
    ...     y_calclass=y_true,
    ...     aim='classification',
    ...     metric='prediction_change_rate'
    ... )
    
    >>> # Exemplo com probability_shift (SVC com probability=True)
    >>> svc_proba = SVC(kernel='rbf', probability=True)
    >>> results = calculate_predicate_perturbation(
    ...     estimator=svc_proba,
    ...     Xcalclass_prep=X_prep,
    ...     folds_struct=folds,
    ...     predicates_df=predicates,
    ...     spectral_cuts=cuts,
    ...     aim='classification',
    ...     metric='probability_shift'
    ... )
    """
    # VALIDAÇÃO DE ENTRADAS
    
    # Verificar se o estimator tem método predict
    if not hasattr(estimator, 'predict'):
        # Lança erro se o modelo não tiver método predict
        raise ValueError(f"O estimator deve ter método predict(). Tipo: {type(estimator)}")
    
    # Verificar se folds_struct é dicionário
    if not isinstance(folds_struct, dict):
        # Lança erro se a estrutura de folds não for dicionário
        raise TypeError("folds_struct deve ser um dicionário")
    
    # Verificar colunas obrigatórias em predicates_df
    required_cols = ['rule', 'zone']  # Colunas mínimas necessárias
    missing_cols = [c for c in required_cols if c not in predicates_df.columns]
    if missing_cols:
        # Lança erro se faltar alguma coluna obrigatória
        raise KeyError(f"Colunas faltando em predicates_df: {missing_cols}")
    
    # Validar aim
    valid_aims = {'regression', 'classification'}
    if aim not in valid_aims:
        raise ValueError(f"aim deve ser um de {valid_aims}. Recebido: {aim}")
    
    # Definir métricas válidas para cada aim
    regression_metrics = {'mean_abs_diff', 'mean_diff', 'mean_relative_dev'}
    classification_metrics = {
        'prediction_change_rate', 
        'accuracy_drop', 
        'f1_drop',
        'probability_shift', 
        'decision_function_shift'
    }
    
    # Validar métrica de acordo com aim
    if aim == 'regression':
        if metric not in regression_metrics:
            raise ValueError(
                f"Para aim='regression', metric deve ser um de {regression_metrics}. "
                f"Recebido: '{metric}'"
            )
    else:  # classification
        if metric not in classification_metrics:
            raise ValueError(
                f"Para aim='classification', metric deve ser um de {classification_metrics}. "
                f"Recebido: '{metric}'"
            )
        
        # Verificar requisitos específicos de cada métrica de classificação
        if metric == 'probability_shift':
            if not hasattr(estimator, 'predict_proba'):
                raise ValueError(
                    f"A métrica 'probability_shift' requer estimator com predict_proba(). "
                    f"Tipo recebido: {type(estimator)}. "
                    f"Dica: para SVC, use SVC(probability=True)"
                )
        
        if metric == 'decision_function_shift':
            if not hasattr(estimator, 'decision_function'):
                raise ValueError(
                    f"A métrica 'decision_function_shift' requer estimator com decision_function(). "
                    f"Tipo recebido: {type(estimator)}. "
                    f"Modelos compatíveis: SVC, LinearSVC, LogisticRegression, etc."
                )
        
        if metric in ['accuracy_drop', 'f1_drop']:
            if y_calclass is None:
                raise ValueError(
                    f"A métrica '{metric}' requer y_calclass (rótulos verdadeiros). "
                    f"Forneça y_calclass como parâmetro."
                )
    
    # Converter y_calclass para Series se necessário
    if y_calclass is not None:
        if isinstance(y_calclass, np.ndarray):
            y_calclass = pd.Series(y_calclass)
    
    # INICIALIZAÇÃO
    
    # Dicionário para armazenar resultados finais (compatível com pipeline existente)
    metrics_results_dict = {}
    
    # Dicionário para armazenar resultados detalhados
    detailed_results = {}
    
    # Nome da coluna de métrica no DataFrame de saída
    metric_name = 'Perturbation'
    
    # Contadores para estatísticas
    total_folds = len(folds_struct)  # Total de folds a processar
    total_predicates_processed = 0   # Contador de predicados processados
    total_predicates_skipped = 0     # Contador de predicados ignorados
    
    # Validar perturbation_mode
    valid_modes = {'constant', 'mean', 'median', 'min', 'max'}
    if perturbation_mode not in valid_modes:
        raise ValueError(f"perturbation_mode deve ser um de {valid_modes}. Recebido: {perturbation_mode}")
    
    # Validar stats_source
    valid_sources = {'full', 'predicate'}
    if stats_source not in valid_sources:
        raise ValueError(f"stats_source deve ser um de {valid_sources}. Recebido: {stats_source}")
    
    # Log inicial se verbose
    if verbose:
        print("=" * 70)
        print("PERTURBATION IMPORTANCE PARA PREDICADOS")
        print("=" * 70)
        print(f"Tipo de tarefa (aim): {aim}")
        print(f"Modo de perturbação: {perturbation_mode}")
        if perturbation_mode == 'constant':
            print(f"Valor de perturbação: {perturbation_value}")
        else:
            print(f"Fonte das estatísticas: {stats_source}")
        print(f"Métrica: {metric}")
        print(f"Total de folds: {total_folds}")
        if aim == 'classification' and y_calclass is not None:
            print(f"Classes em y_calclass: {y_calclass.unique().tolist()}")
        print()
    
    # LOOP PRINCIPAL: PROCESSAR CADA FOLD
    
    # Iterar sobre cada fold na estrutura
    for fold_idx, (fold_name, predicates_dict) in enumerate(folds_struct.items()):
        
        # Log do fold atual
        if verbose:
            print(f"\n[{fold_name}] Processando {len(predicates_dict)} predicados...")
        
        # Verificar se o fold está vazio
        if len(predicates_dict) == 0:
            # Se vazio, criar DataFrame vazio e pular para próximo fold
            if verbose:
                print(f"  VAZIO - pulando")
            metrics_results_dict[fold_name] = pd.DataFrame({
                'Predicate': [],
                metric_name: []
            })
            continue
        
        # Dicionário temporário para métricas deste fold
        fold_metrics = {}
        
        # Dicionário temporário para resultados detalhados deste fold
        fold_detailed = {}
        
        # LOOP: PROCESSAR CADA PREDICADO NO FOLD
        
        # Iterar sobre cada predicado do fold
        for pred_rule, df_info in predicates_dict.items():
            
            # Incrementar contador de predicados processados
            total_predicates_processed += 1
            
            # 1. OBTER ÍNDICES DE AMOSTRAS DO PREDICADO
            
            # Extrair índices das amostras que pertencem a este predicado
            sample_indices = df_info['Sample_Index'].values.tolist()
            
            # Número de amostras no predicado
            n_samples = len(sample_indices)
            
            # Log do predicado atual
            if verbose:
                print(f"  Predicado: {pred_rule} (n={n_samples})")
            
            # 2. VERIFICAR CASOS LIMITES
            
            # Se não há amostras, não é possível calcular importância
            if n_samples == 0:
                if verbose:
                    print(f"    SKIP: n_samples=0 (sem amostras)")
                # Atribuir importância zero
                fold_metrics[pred_rule] = 0.0
                # Salvar detalhes
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': 'n_samples = 0'
                }
                # Incrementar contador de skips
                total_predicates_skipped += 1
                continue
            
            # 3. OBTER INFORMAÇÕES DA ZONA ESPECTRAL
            
            # Tentar obter colunas da zona espectral do predicado
            try:
                # Usar função auxiliar para obter colunas da zona
                zone_cols = get_zone_columns_from_predicate(
                    predicate_rule=pred_rule,
                    predicates_df=predicates_df,
                    spectral_cuts=spectral_cuts,
                    Xcal_columns=Xcalclass_prep.columns
                )
            except (KeyError, ValueError) as e:
                # Se erro ao obter zona, atribuir importância zero
                if verbose:
                    print(f"    ERRO ao obter zona: {e}")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': str(e)
                }
                total_predicates_skipped += 1
                continue
            
            # Verificar se a zona tem colunas
            if len(zone_cols) == 0:
                # Se zona vazia, atribuir importância zero
                if verbose:
                    print(f"    SKIP: zona espectral vazia")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': 'zona vazia'
                }
                total_predicates_skipped += 1
                continue
            
            # Log das colunas da zona
            if verbose:
                print(f"    Zona: {len(zone_cols)} colunas")
            
            # 4. OBTER LIMITES DA ZONA PARA PERTURBAÇÃO
            
            # Encontrar nome da zona associada ao predicado
            mask_pred = predicates_df['rule'] == pred_rule
            zone_name = predicates_df.loc[mask_pred, 'zone'].values[0]
            
            # Encontrar limites (start, end) da zona nos spectral_cuts
            zone_start, zone_end = None, None
            for cut in spectral_cuts:
                # Extrair nome e limites do cut
                if len(cut) == 3:
                    name, start, end = cut
                elif len(cut) == 2:
                    start, end = cut
                    name = f"{start}-{end}"
                else:
                    continue
                
                # Verificar se é a zona correta
                if name == zone_name:
                    zone_start, zone_end = float(start), float(end)
                    break
            
            # Se não encontrou limites, pular
            if zone_start is None or zone_end is None:
                if verbose:
                    print(f"    SKIP: limites da zona não encontrados")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': zone_cols,
                    'skip_reason': 'limites não encontrados'
                }
                total_predicates_skipped += 1
                continue
            
            # 5. EXTRAIR DADOS DAS AMOSTRAS DO PREDICADO
            
            # Extrair subconjunto de dados para as amostras do predicado
            X_eval = Xcalclass_prep.iloc[sample_indices].copy()
            
            # Extrair rótulos verdadeiros se disponíveis (para métricas de classificação)
            if y_calclass is not None:
                y_true_eval = y_calclass.iloc[sample_indices]
            else:
                y_true_eval = None
            
            # 6. PERTURBAR ZONA ESPECTRAL
            
            # Criar cópia dos dados para perturbação
            X_perturbed = X_eval.copy()
            
            # Aplicar perturbação de acordo com o modo escolhido
            if perturbation_mode == 'constant':
                # Comportamento original: valor fixo para todas as colunas
                X_perturbed[zone_cols] = perturbation_value
            else:
                # Calcular estatísticas por coluna
                # Escolher fonte dos dados para estatísticas
                if stats_source == 'full':
                    stats_data = Xcalclass_prep[zone_cols]
                else:  # 'predicate'
                    stats_data = X_eval[zone_cols]
                
                # Calcular estatística de acordo com o modo
                if perturbation_mode == 'mean':
                    col_stats = stats_data.mean(axis=0)
                elif perturbation_mode == 'median':
                    col_stats = stats_data.median(axis=0)
                elif perturbation_mode == 'min':
                    col_stats = stats_data.min(axis=0)
                elif perturbation_mode == 'max':
                    col_stats = stats_data.max(axis=0)
                
                # Substituir cada coluna pela sua estatística
                for col in zone_cols:
                    X_perturbed[col] = col_stats[col]
            
            # 7. CALCULAR IMPORTÂNCIA BASEADA NO AIM E MÉTRICA ESCOLHIDA
            
            if aim == 'regression':
                # =====================================================================
                # MODO REGRESSÃO: usa predict() e métricas numéricas contínuas
                # =====================================================================
                
                # Fazer predição com dados originais
                y_pred_original = estimator.predict(X_eval)
                y_pred_original = np.array(y_pred_original).flatten()
                
                # Fazer predição com dados perturbados
                y_pred_perturbed = estimator.predict(X_perturbed)
                y_pred_perturbed = np.array(y_pred_perturbed).flatten()
                
                # Calcular importância de acordo com a métrica
                if metric == 'mean_abs_diff':
                    # Média da diferença absoluta entre predições
                    importance = np.mean(np.abs(y_pred_original - y_pred_perturbed))
                elif metric == 'mean_diff':
                    # Média da diferença (com sinal)
                    importance = np.mean(y_pred_original - y_pred_perturbed)
                elif metric == 'mean_relative_dev':
                    # Média do desvio relativo (cuidado com divisão por zero)
                    y_safe = np.where(y_pred_original == 0, np.nan, y_pred_original)
                    rel_dev = (y_pred_perturbed - y_pred_original) / y_safe
                    importance = np.nanmean(rel_dev)
                
                # Para ranking, usar valor absoluto para métricas com sinal
                if metric in ['mean_diff', 'mean_relative_dev']:
                    importance_for_ranking = np.abs(importance)
                else:
                    importance_for_ranking = importance
                    
            else:  # aim == 'classification'
                # =====================================================================
                # MODO CLASSIFICAÇÃO: usa predict(), predict_proba() ou decision_function()
                # =====================================================================
                
                if metric == 'prediction_change_rate':
                    # -----------------------------------------------------------------
                    # PREDICTION CHANGE RATE: proporção de amostras que mudaram de classe
                    # Usa: estimator.predict()
                    # -----------------------------------------------------------------
                    
                    # Fazer predição com dados originais
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_original = np.array(y_pred_original).flatten()
                    
                    # Fazer predição com dados perturbados
                    y_pred_perturbed = estimator.predict(X_perturbed)
                    y_pred_perturbed = np.array(y_pred_perturbed).flatten()
                    
                    # Calcular proporção de amostras que mudaram de classe
                    # Valores de 0 a 1, onde 1 = todas mudaram
                    importance = np.mean(y_pred_original != y_pred_perturbed)
                    importance_for_ranking = importance
                    
                elif metric == 'accuracy_drop':
                    # -----------------------------------------------------------------
                    # ACCURACY DROP: queda na acurácia após perturbação
                    # Usa: estimator.predict() + y_true
                    # -----------------------------------------------------------------
                    
                    # Fazer predição com dados originais
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_original = np.array(y_pred_original).flatten()
                    
                    # Fazer predição com dados perturbados
                    y_pred_perturbed = estimator.predict(X_perturbed)
                    y_pred_perturbed = np.array(y_pred_perturbed).flatten()
                    
                    # Calcular acurácia original e após perturbação
                    acc_original = accuracy_score(y_true_eval, y_pred_original)
                    acc_perturbed = accuracy_score(y_true_eval, y_pred_perturbed)
                    
                    # Queda na acurácia (positivo = piora)
                    importance = acc_original - acc_perturbed
                    importance_for_ranking = np.abs(importance)
                    
                elif metric == 'f1_drop':
                    # -----------------------------------------------------------------
                    # F1 DROP: queda no F1-score após perturbação
                    # Usa: estimator.predict() + y_true
                    # -----------------------------------------------------------------
                    
                    # Fazer predição com dados originais
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_original = np.array(y_pred_original).flatten()
                    
                    # Fazer predição com dados perturbados
                    y_pred_perturbed = estimator.predict(X_perturbed)
                    y_pred_perturbed = np.array(y_pred_perturbed).flatten()
                    
                    # Calcular F1 original e após perturbação
                    # Usa average='weighted' para suportar multiclasse
                    f1_original = f1_score(y_true_eval, y_pred_original, average='weighted')
                    f1_perturbed = f1_score(y_true_eval, y_pred_perturbed, average='weighted')
                    
                    # Queda no F1 (positivo = piora)
                    importance = f1_original - f1_perturbed
                    importance_for_ranking = np.abs(importance)
                    
                elif metric == 'probability_shift':
                    # -----------------------------------------------------------------
                    # PROBABILITY SHIFT: diferença nas probabilidades preditas
                    # Usa: estimator.predict_proba()
                    # -----------------------------------------------------------------
                    
                    # Obter probabilidades originais
                    prob_original = estimator.predict_proba(X_eval)
                    
                    # Obter probabilidades após perturbação
                    prob_perturbed = estimator.predict_proba(X_perturbed)
                    
                    # Calcular diferença nas probabilidades
                    # IMPORTANTE: Para classificação, predict_proba retorna (n_samples, n_classes)
                    # onde cada linha soma 1.0. Para evitar contar mudanças redundantes,
                    # calculamos a mudança POR AMOSTRA (soma das diferenças absolutas por linha)
                    # e depois fazemos a média entre amostras.
                    #
                    # Exemplo binário: [0.7, 0.3] → [0.6, 0.4]
                    # - Sem correção: mean(|0.7-0.6| + |0.3-0.4|) = mean(0.1 + 0.1) = 0.2 ❌
                    # - Com correção: mean(|0.7-0.6| + |0.3-0.4|) / 2 = 0.1 ✓
                    #
                    # Para k classes, dividimos por k para normalizar e obter valores
                    # comparáveis entre problemas binários e multiclasse.
                    
                    n_classes = prob_original.shape[1]
                    
                    # Calcular mudança total por amostra (soma sobre classes)
                    shift_per_sample = np.sum(np.abs(prob_original - prob_perturbed), axis=1)
                    
                    # Normalizar pelo número de classes (evita contar mudanças redundantes)
                    # Dividir por 2 porque mudanças em probabilidades que somam 1 são simétricas
                    shift_per_sample_normalized = shift_per_sample / 2.0
                    
                    # Média sobre todas as amostras
                    importance = np.mean(shift_per_sample_normalized)
                    importance_for_ranking = importance
                    
                    # Para verbose, salvar as predições de classe também
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_perturbed = estimator.predict(X_perturbed)
                    
                elif metric == 'decision_function_shift':
                    # -----------------------------------------------------------------
                    # DECISION FUNCTION SHIFT: diferença nos valores da decision function
                    # Usa: estimator.decision_function()
                    # Útil para SVM e modelos lineares
                    # -----------------------------------------------------------------
                    
                    # Obter valores da decision function originais
                    df_original = estimator.decision_function(X_eval)
                    df_original = np.array(df_original)
                    
                    # Achatar se necessário (para classificação binária)
                    if df_original.ndim == 1:
                        df_original = df_original.flatten()
                    
                    # Obter valores da decision function após perturbação
                    df_perturbed = estimator.decision_function(X_perturbed)
                    df_perturbed = np.array(df_perturbed)
                    
                    # Achatar se necessário
                    if df_perturbed.ndim == 1:
                        df_perturbed = df_perturbed.flatten()
                    
                    # Calcular média da diferença absoluta
                    importance = np.mean(np.abs(df_original - df_perturbed))
                    importance_for_ranking = importance
                    
                    # Para verbose, salvar as predições de classe também
                    y_pred_original = estimator.predict(X_eval)
                    y_pred_perturbed = estimator.predict(X_perturbed)
            
            # 8. ARMAZENAR RESULTADOS
            
            # Armazenar importância para ranking
            fold_metrics[pred_rule] = importance_for_ranking
            
            # Salvar detalhes completos
            fold_detailed[pred_rule] = {
                'importance': importance,
                'importance_abs': np.abs(importance) if isinstance(importance, (int, float)) else importance,
                'n_samples': n_samples,
                'zone_columns': zone_cols,
                'n_zone_features': len(zone_cols),
                'zone_name': zone_name,
                'zone_start': zone_start,
                'zone_end': zone_end,
                'perturbation_mode': perturbation_mode,
                'stats_source': stats_source if perturbation_mode != 'constant' else None,
                'aim': aim,
                'metric': metric
            }
            
            # Log da importância calculada
            if verbose:
                print(f"    Importance: {importance:.6f}")
        
        # CONVERTER PARA DATAFRAME (compatível com pipeline existente)
        
        # Criar DataFrame a partir do dicionário de métricas
        metrics_df = pd.DataFrame.from_dict(
            fold_metrics,
            orient='index',
            columns=[metric_name]
        )
        
        # Adicionar coluna de predicado
        metrics_df.insert(0, 'Predicate', metrics_df.index)
        
        # Resetar índice para ter índice numérico
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Ordenar de forma DECRESCENTE (maiores valores = mais importantes)
        metrics_df = metrics_df.sort_values(by=metric_name, ascending=False)
        
        # Resetar índice após ordenação
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Armazenar resultado do fold
        metrics_results_dict[fold_name] = metrics_df
        
        # Armazenar resultados detalhados do fold
        detailed_results[fold_name] = fold_detailed
    
    # RESUMO FINAL
    
    # Imprimir resumo se verbose
    if verbose:
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        print(f"Tipo de tarefa (aim): {aim}")
        print(f"Métrica utilizada: {metric}")
        print(f"Folds processados: {total_folds}")
        print(f"Predicados processados: {total_predicates_processed}")
        print(f"Predicados ignorados: {total_predicates_skipped}")
        print()
        # Mostrar resumo por fold
        for fold_name, df in metrics_results_dict.items():
            # Ignorar chave especial de resultados detalhados
            if fold_name.startswith('__'):
                continue
            print(f"  {fold_name}: {len(df)} predicados")
    
    # SALVAR RESULTADOS DETALHADOS (OPCIONAL)
    
    # Se solicitado, criar DataFrame com todos os detalhes
    if save_detailed_results:
        # Lista para armazenar linhas do DataFrame detalhado
        detailed_rows = []
        
        # Iterar sobre folds e predicados
        for fold_name, fold_data in detailed_results.items():
            for pred_rule, pred_data in fold_data.items():
                # Adicionar linha com informações do predicado
                detailed_rows.append({
                    'fold': fold_name,
                    'predicate': pred_rule,
                    'importance': pred_data['importance'],
                    'importance_abs': pred_data.get('importance_abs', np.abs(pred_data['importance'])),
                    'n_samples': pred_data['n_samples'],
                    'n_zone_features': pred_data.get('n_zone_features', 0),
                    'zone_name': pred_data.get('zone_name', None),
                    'skip_reason': pred_data.get('skip_reason', None),
                    'aim': pred_data.get('aim', aim),
                    'metric': pred_data.get('metric', metric)
                })
        
        # Criar DataFrame de resultados detalhados
        detailed_df = pd.DataFrame(detailed_rows)
        
        # Anexar como chave especial no dicionário de resultados
        metrics_results_dict['__detailed_perturbation_results__'] = detailed_df
    
    # Retornar dicionário com resultados
    return metrics_results_dict


def map_thresholds_to_natural(
    lrc_df,                    # DataFrame com Zone e Threshold como colunas (espaço pré-processado)
    zone_sums_preprocessed,    # zone_sums_df (pré-processado)
    zone_sums_natural          # zone_sums_df_original (natural)
):
    """
    Mapeia thresholds do espaço pré-processado para o espaço natural
    usando a amostra mais próxima como referência.

    Returns:
        DataFrame com colunas adicionais: 'Threshold_Natural', 'Sample_Index', 'Approximation_Error', 'Node', 'Operator', 'Node_Natural'
    """
    result_df = lrc_df.copy()

    natural_thresholds = []
    sample_indices = []
    approximation_errors = []
    node_natural_list = []

    for idx, row in result_df.iterrows():
        zone_name = row['Zone']
        threshold_val = row['Threshold']
        operator = row['Operator']
        node = row['Node']

        # Skip None values
        if zone_name is None or threshold_val is None or zone_name not in zone_sums_preprocessed.columns:
            natural_thresholds.append(None)
            sample_indices.append(None)
            approximation_errors.append(None)
            node_natural_list.append(None)
            continue

        threshold = float(threshold_val)

        # Encontrar índice da amostra mais próxima no espaço pré-processado
        zone_values_prep = zone_sums_preprocessed[zone_name]
        distances = (zone_values_prep - threshold).abs()
        closest_idx = distances.idxmin()  # índice da amostra mais próxima

        # Buscar valor correspondente no espaço natural
        natural_value = zone_sums_natural.loc[closest_idx, zone_name]

        # Calcular erro de aproximação (no espaço pré-processado)
        error = distances.loc[closest_idx]

        # Montar Node_Natural (ex: "Zone > 0.123")
        if operator is not None and natural_value is not None:
            node_natural = f"{zone_name} {operator} {natural_value:.6f}"
        else:
            node_natural = None

        natural_thresholds.append(natural_value)
        sample_indices.append(closest_idx)
        approximation_errors.append(error)
        node_natural_list.append(node_natural)

    result_df['Threshold_Natural'] = natural_thresholds
    result_df['Reference_Sample_Index'] = sample_indices
    result_df['Approximation_Error'] = approximation_errors
    result_df['Node'] = lrc_df.get('Node')
    result_df['Operator'] = lrc_df.get('Operator')
    result_df['Node_Natural'] = node_natural_list

    return result_df

def aggregate_spectral_zones_pca(spectral_zones_dict):
    """
    Agrega zonas espectrais usando PCA com 1 componente principal.
    
    Para cada zona espectral, ajusta uma PCA com 1 componente e extrai:
    - Scores: projeção das amostras na direção de máxima variância
    - Loadings: pesos de cada variável na PC1
    - Média: vetor de médias da zona (para reconstrução)
    - Variância Explicada: fração da variância capturada pela PC1
    
    Parameters
    ----------
    spectral_zones_dict : dict
        Dicionário retornado por extract_spectral_zones.
        Chaves = nomes das zonas, Valores = DataFrames com dados espectrais.
    
    Returns
    -------
    scores_df : pd.DataFrame
        DataFrame com scores da PC1 para cada zona (amostras x zonas).
    pca_info_dict : dict
        Dicionário com informações da PCA para cada zona:
        - 'loadings': vetor de loadings da PC1
        - 'mean': vetor de médias da zona
        - 'variance_explained': fração de variância explicada
        - 'columns': nomes das colunas originais (para reconstrução)
    """
    from sklearn.decomposition import PCA
    import pandas as pd

    scores_dict = {}  # armazena scores de cada zona
    pca_info_dict = {}  # armazena informações para reconstrução
    
    for zone_name, zone_df in spectral_zones_dict.items():
        # 1: Preparação dos dados
        X_zone = zone_df.values  # converter para numpy array
        
        # 2: Ajuste da PCA com 1 componente
        pca = PCA(n_components=1)
        scores = pca.fit_transform(X_zone)  # scores da PC1 (n_samples, 1)
        
        # 3: Extração das informações
        loadings = pca.components_[0]  # loadings da PC1 (d_m,)
        mean_vector = pca.mean_  # vetor de médias (d_m,)
        variance_explained = pca.explained_variance_ratio_[0]  # fração de variância
        
        # 4: Armazenamento
        scores_dict[zone_name] = scores.flatten()  # converter para 1D
        
        pca_info_dict[zone_name] = {
            'loadings': loadings,
            'mean': mean_vector,
            'variance_explained': variance_explained,
            'columns': zone_df.columns.tolist(),  # nomes das colunas originais
            'pca_model': pca  # modelo PCA completo (para uso futuro)
        }
        
        # Log informativo
        print(f"Zona '{zone_name}': VE = {variance_explained:.2%}, "
              f"dim = {len(loadings)} variáveis")
    
    # Criar DataFrame com todos os scores
    scores_df = pd.DataFrame(scores_dict)
    
    return scores_df, pca_info_dict

def reconstruct_threshold_to_spectrum(threshold_value, zone_name, pca_info_dict):
    """
    Reconstrói um threshold escalar (no espaço dos scores) para o espaço 
    espectral original, gerando um "espectro de threshold" multivariado.
    
    Fórmula matemática:
        τ = mean + threshold_value * loadings
    
    Parameters
    ----------
    threshold_value : float
        Valor do threshold no espaço dos scores da PC1.
    zone_name : str
        Nome da zona espectral.
    pca_info_dict : dict
        Dicionário com informações da PCA (retornado por aggregate_spectral_zones_pca).
    
    Returns
    -------
    threshold_spectrum : pd.Series
        Espectro de threshold com índice = energias/comprimentos de onda originais.
    """
    import pandas as pd
    # Recuperar informações da PCA
    pca_info = pca_info_dict[zone_name]
    loadings = pca_info['loadings']
    mean_vector = pca_info['mean']
    columns = pca_info['columns']
    
    # Reconstrução: τ = mean + q * loadings
    threshold_spectrum = mean_vector + threshold_value * loadings
    
    # Converter para Series com índice original
    threshold_spectrum = pd.Series(threshold_spectrum, index=columns, name=f'threshold_{threshold_value:.4f}')
    
    return threshold_spectrum

def extract_predicate_info(predicate_rule):
    """
    Extrai informações de uma regra de predicado.
    
    Parameters
    ----------
    predicate_rule : str
        Regra no formato "zone_name <= threshold" ou "zone_name > threshold"
    
    Returns
    -------
    dict : {'zone': str, 'operator': str, 'threshold': float}
    """
    if '<=' in predicate_rule:
        parts = predicate_rule.split('<=')
        operator = '<='
    elif '>' in predicate_rule:
        parts = predicate_rule.split('>')
        operator = '>'
    else:
        raise ValueError(f"Operador não reconhecido em: {predicate_rule}")
    
    zone_name = parts[0].strip()
    threshold_value = float(parts[1].strip())
    
    return {
        'zone': zone_name,
        'operator': operator,
        'threshold': threshold_value
    }

def extract_zone_from_predicate(predicate_rule):
    """
    Extrai o nome da zona espectral a partir da regra do predicado.
    
    Parameters
    ----------
    predicate_rule : str
        Regra no formato "zone_name <= threshold" ou "zone_name > threshold"
    
    Returns
    -------
    str : Nome da zona espectral
    
    Examples
    --------
    >>> extract_zone_from_predicate("Ca ka <= 25.5")
    'Ca ka'
    >>> extract_zone_from_predicate("Fe ka > 10.2")
    'Fe ka'
    """
    if '<=' in predicate_rule:
        return predicate_rule.split('<=')[0].strip()
    elif '>' in predicate_rule:
        return predicate_rule.split('>')[0].strip()
    else:
        raise ValueError(f"Operador não reconhecido em: {predicate_rule}")


def build_predicate_graph(bags_result, predicate_ranking_dict, 
                            metric_column='Cov',
                            random_state=42, show_details=True,
                            var_exp=False, pca_info_dict=None):
    """
    Constrói um grafo direcionado de predicados onde os pesos das arestas
    são baseados na Covariância (ou outra métrica) do predicado de ORIGEM.
    
    Parameters
    ----------
    - **bags_result** : dict
        Dicionário com bags de predicados:
        {'Bag_1': {'Ca ka <= 25.5': DataFrame, ...}, 'Bag_2': {...}, ...}
        
    - **predicate_ranking_dict** : dict
        Dicionário com rankings de predicados segundo uma métrica para cada bag:
        {'Bag_1': DataFrame(['Predicate', metric_column]), 'Bag_2': ...}
        
    - **metric_column** : str, default='Cov'
        Nome da coluna no predicate_ranking_dict que contém a métrica de ordenação.
        Permite flexibilidade para usar 'Cov', 'Permutation', etc.
        
    - **random_state** : int, default=42
        Semente para desempate aleatório de arestas bidirecionais.
        
    - **show_details** : bool, default=True
        Se True, imprime detalhes sobre remoção de arestas bidirecionais.
        
    - **var_exp** : bool, default=False
        Se True, multiplica os pesos das arestas pela variância explicada (PC1)
        da zona espectral correspondente ao predicado de origem.
        
    - **pca_info_dict** : dict, optional
        Dicionário com informações da PCA para cada zona (obrigatório se var_exp=True).
        Chaves = nomes das zonas, Valores = dict com 'variance_explained'.
    
    Returns
    -------
    - **DG** : nx.DiGraph
        Grafo direcionado com pesos baseados na métrica acumulada.

    """
    import networkx as nx
    import numpy as np
    import pandas as pd
    
    # Validação dos parâmetros var_exp
    if var_exp:
        if pca_info_dict is None:
            raise ValueError("pca_info_dict é obrigatório quando var_exp=True")
    
    # Define semente para reprodutibilidade nos desempates
    np.random.seed(random_state)
    
    # FASE 1: INICIALIZAÇÃO DO GRAFO
    DG = nx.DiGraph()
    DG.add_node('Class_A', node_type='terminal', class_label='A')
    DG.add_node('Class_B', node_type='terminal', class_label='B')
    
    # FASE 2: CONSTRUÇÃO DOS CAMINHOS E ACUMULAÇÃO DE PESOS
    for bag_name, bag_predicates_dict in bags_result.items():
        
        # 2.1: Obtém o ranking de métricas para este bag
        predicate_ranking = predicate_ranking_dict[bag_name]
        ordered_predicates = predicate_ranking['Predicate'].tolist()
        
        # Filtra apenas predicados que existem neste bag específico
        ordered_predicates = [p for p in ordered_predicates if p in bag_predicates_dict.keys()]
        
        if len(ordered_predicates) == 0:
            continue
        
        # 2.2: Cria dicionário de lookup para métrica
        ranking_lookup = dict(zip(predicate_ranking['Predicate'], predicate_ranking[metric_column]))
        
        # 2.3: Constrói arestas entre predicados consecutivos
        for i in range(len(ordered_predicates) - 1):
            pred_current = ordered_predicates[i]
            pred_next = ordered_predicates[i + 1]
            
            DG.add_node(pred_current, node_type='predicate')
            DG.add_node(pred_next, node_type='predicate')
            
            ranking_value = float(ranking_lookup[pred_current])
            
            # Ponderar pela variância explicada se var_exp=True
            if var_exp:
                zone_name = extract_zone_from_predicate(pred_current)
                if zone_name in pca_info_dict:
                    ranking_value *= pca_info_dict[zone_name]['variance_explained']
            
            # Acumular peso se aresta já existe
            if DG.has_edge(pred_current, pred_next):
                DG[pred_current][pred_next]['weight'] += ranking_value
            else:
                DG.add_edge(pred_current, pred_next, weight=ranking_value, bag=bag_name)
        
        # 2.4: Conecta o ÚLTIMO predicado ao nó terminal
        last_pred = ordered_predicates[-1]
        DG.add_node(last_pred, node_type='predicate')
        
        df_last = bag_predicates_dict[last_pred]
        class_counts = df_last['Class_Predicted'].value_counts()
        majority_class = class_counts.idxmax()
        terminal_node = f'Class_{majority_class}'
        
        ranking_last_value = float(ranking_lookup[last_pred])
        
        if var_exp:
            zone_name = extract_zone_from_predicate(last_pred)
            if zone_name in pca_info_dict:
                ranking_last_value *= pca_info_dict[zone_name]['variance_explained']
        
        if DG.has_edge(last_pred, terminal_node):
            DG[last_pred][terminal_node]['weight'] += ranking_last_value
        else:
            DG.add_edge(last_pred, terminal_node, weight=ranking_last_value, bag=bag_name)

    # FASE 3: IDENTIFICAÇÃO DE ARESTAS BIDIRECIONAIS
    bidirectional_pairs = []
    processed = set()
    
    for u, v in DG.edges():
        if DG.has_edge(v, u) and (v, u) not in processed:
            bidirectional_pairs.append({
                'node_A': u, 'node_B': v,
                'weight_A_to_B': float(DG[u][v]['weight']),
                'weight_B_to_A': float(DG[v][u]['weight'])
            })
            processed.add((u, v))
            processed.add((v, u))
    
    print(f"\nTotal de pares bidirecionais encontrados: {len(bidirectional_pairs)}")
    
    # FASE 4: RESOLUÇÃO DE ARESTAS BIDIRECIONAIS
    n_removed = 0
    
    for pair in bidirectional_pairs:
        u, v = pair['node_A'], pair['node_B']
        w_fwd, w_rev = pair['weight_A_to_B'], pair['weight_B_to_A']
        
        if w_fwd > w_rev:
            DG.remove_edge(v, u)
            if show_details:
                print(f"Removida: {v} -> {u} ({w_rev:.4f}) | Mantida: {u} -> {v} ({w_fwd:.4f})")
        elif w_rev > w_fwd:
            DG.remove_edge(u, v)
            if show_details:
                print(f"Removida: {u} -> {v} ({w_fwd:.4f}) | Mantida: {v} -> {u} ({w_rev:.4f})")
        else:
            # Empate: escolha aleatória
            if np.random.rand() > 0.5:
                DG.remove_edge(v, u)
                if show_details:
                    print(f"Empate! Removida: {v} -> {u} ({w_rev:.4f})")
            else:
                DG.remove_edge(u, v)
                if show_details:
                    print(f"Empate! Removida: {u} -> {v} ({w_fwd:.4f})")
        n_removed += 1

    # FASE 5: RESUMO FINAL
    print(f"\n{'='*70}")
    print("RESUMO DO GRAFO CONSTRUÍDO")
    print(f"{'='*70}")
    print(f"Arestas iniciais: {DG.number_of_edges() + n_removed} | Removidas: {n_removed}")
    print(f"Nós predicados: {len([n for n, attr in DG.nodes(data=True) if attr['node_type'] == 'predicate'])}")
    print(f"Métrica: {metric_column}")
    if var_exp:
        print("Ponderação por variância explicada: ATIVADA")
    
    return DG