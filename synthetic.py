import numpy as np
import pandas as pd

def modelo_pico_gaussiano(x, centro, amplitude, largura):
    """
    Gera um pico Gaussiano unidimensional.
    
    A função implementa a equação:
    g(x) = A * exp(-(x - c)² / (2σ²))
    
    Parâmetros
    ----------
    x : array_like
        Eixo espectral (comprimentos de onda, energia, canais)
    centro : float
        Posição central do pico (mesma unidade de x)
    amplitude : float
        Altura máxima do pico (intensidade no centro)
    largura : float
        Desvio padrão (σ) do pico — controla a dispersão/largura
        
    Retorna
    -------
    ndarray
        Array com o pico Gaussiano avaliado em cada ponto de x
        
    Notas
    -----
    - Para XRF: usar largura pequena (~5-15) para simular linhas estreitas
    - Para Vis-NIR: usar largura maior (~20-50) para bandas largas
    """
    return amplitude * np.exp(-(x - centro)**2 / (2 * largura**2))

def _gerar_espectro_unico(
    x, 
    centros_picos, 
    amp_media=1.0, 
    amp_std=0.1, 
    larg_media=15.0, 
    larg_std=2.0, 
    ruido_std=0.02
):
    """
    Gera um único espectro somando picos Gaussianos com variabilidade + ruído.
    
    Função auxiliar interna para gerar_dados_espectrais_sinteticos.
    
    Parâmetros
    ----------
    x : ndarray
        Eixo espectral
    centros_picos : list of float
        Posições dos picos a serem adicionados
    amp_media, amp_std : float
        Média e desvio padrão da amplitude dos picos
    larg_media, larg_std : float
        Média e desvio padrão da largura dos picos
    ruido_std : float
        Desvio padrão do ruído gaussiano de linha de base
        
    Retorna
    -------
    ndarray
        Espectro sintético (linha de base + picos)
    """
    # Linha de base: ruído branco gaussiano
    espectro = np.random.normal(0, ruido_std, len(x))
    
    # Adicionar cada pico com variabilidade aleatória
    for centro in centros_picos:
        amp = np.random.normal(amp_media, amp_std)
        larg = np.random.normal(larg_media, larg_std)
        espectro += modelo_pico_gaussiano(x, centro, amp, larg)
    
    return espectro

# FUNÇÃO PRINCIPAL (MULTI-CLASSE)

def generate_synthetic_spectral_data(
    configuracao_classes,
    n_pontos=500,
    x_min=0,
    x_max=1000,
    seed=None
):
    """
    Gera conjunto de espectros sintéticos para múltiplas classes (N classes).
    
    Retorna um DataFrame onde:
    - Primeira coluna: 'classe' (valores definidos pelo usuário: 'A', 'B', 'C', ...)
    - Demais colunas: variáveis espectrais (valores de intensidade)
    - Linhas: amostras individuais
    
    Parâmetros
    ----------
    configuracao_classes : list of dict
        Lista de dicionários, onde cada dicionário define uma classe com:
        - 'nome': str, nome da classe (ex: 'A', 'B', 'Classe1', etc.)
        - 'n_amostras': int, número de amostras para esta classe
        - 'picos': list of float, posições dos picos no eixo espectral
        - 'amp_media': float, optional (default=1.0), média da amplitude dos picos
        - 'amp_std': float, optional (default=0.1), desvio padrão da amplitude
        - 'larg_media': float, optional (default=15.0), média da largura dos picos
        - 'larg_std': float, optional (default=2.0), desvio padrão da largura
        - 'ruido_std': float, optional (default=0.02), desvio padrão do ruído de base
        
        Exemplo:
        [
            {
                'nome': 'A',
                'n_amostras': 50,
                'picos': [250, 550, 700, 850],
                'amp_media': 1.0,
                'larg_media': 15.0
            },
            {
                'nome': 'B',
                'n_amostras': 50,
                'picos': [250, 700, 850],
                'amp_media': 1.2,
                'larg_media': 20.0
            }
        ]
        
    n_pontos : int, default=500
        Número de pontos no eixo espectral (resolução)
    x_min, x_max : float, default=0, 1000
        Limites do eixo espectral (ex: 400-1000 nm para Vis-NIR, 0-40 keV para XRF)
    seed : int, optional
        Semente para reprodutibilidade
        
    Retorna
    -------
    df : pandas.DataFrame
        DataFrame com espectros sintéticos.
        - Coluna 0: 'classe' (str: nome definido em configuracao_classes)
        - Colunas 1 a n_pontos: variáveis espectrais com nomes baseados no eixo x
        - Shape: (total_amostras, n_pontos + 1)
     """
    # Configurar semente para reprodutibilidade
    if seed is not None:
        np.random.seed(seed)
    
    # Criar eixo espectral
    x_eixo = np.linspace(x_min, x_max, n_pontos)
    
    # Listas para armazenar espectros e labels
    espectros_lista = []
    classes_lista = []
    
    # Iterar sobre cada configuração de classe
    for config in configuracao_classes:
        # Extrair parâmetros da classe (com valores padrão)
        nome_classe = config['nome']
        n_amostras = config['n_amostras']
        picos = config['picos']
        amp_media = config.get('amp_media', 1.0)
        amp_std = config.get('amp_std', 0.1)
        larg_media = config.get('larg_media', 15.0)
        larg_std = config.get('larg_std', 2.0)
        ruido_std = config.get('ruido_std', 0.02)
        
        # Gerar espectros para esta classe
        for _ in range(n_amostras):
            espectro = _gerar_espectro_unico(
                x_eixo, picos, amp_media, amp_std, 
                larg_media, larg_std, ruido_std
            )
            espectros_lista.append(espectro)
            classes_lista.append(nome_classe)
    
    # Converter para array numpy
    espectros_array = np.array(espectros_lista)
    
    # Criar DataFrame: colunas = variáveis espectrais
    nome_colunas = x_eixo.astype(str).tolist()
    df = pd.DataFrame(espectros_array, columns=nome_colunas)
    
    # Inserir coluna 'classe' na posição 0
    df.insert(0, 'Class', classes_lista)
    
    return df