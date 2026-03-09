#Escalonamento de Poisson
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def poisson(input_data, mc=True):
    """
    Poisson + mc
    ----------
    Retorna o dado preprocessado, media no espaço original, media na escala poisson + mc
    """
    mean_original = np.mean(input_data, axis=0)
    escala_poisson = 1 / np.sqrt(mean_original) 
    X_poisson = pd.DataFrame(input_data * escala_poisson)
    X_poisson[np.isnan(X_poisson)] = 0
    if mc:
        mean_poisson = np.mean(X_poisson, axis=0)
        X_poisson_mc = X_poisson - mean_poisson
        X_poisson_mc[np.isnan(X_poisson_mc)] = 0
        X_poisson_mc = pd.DataFrame(X_poisson_mc)
        return X_poisson_mc, mean_original, mean_poisson
    else:
        return X_poisson, mean_original

def modified_poisson(input_data, degre=2, mc=True):
    """
    Poisson + mc
    ----------
    Retorna o dado preprocessado, media no espaço original, media na escala poisson + mc
    """
    mean_original = np.mean(input_data, axis=0)
    escala_poisson = 1 / (mean_original) ** (1//degre) 
    X_poisson = pd.DataFrame(input_data * escala_poisson)
    X_poisson[np.isnan(X_poisson)] = 0
    if mc:
        mean_poisson = np.mean(X_poisson, axis=0)
        X_poisson_mc = X_poisson - mean_poisson
        X_poisson_mc[np.isnan(X_poisson_mc)] = 0
        X_poisson_mc = pd.DataFrame(X_poisson_mc)
        return X_poisson_mc, mean_original, mean_poisson
    else:
        return X_poisson, mean_original

def pareto(input_data, mc=True):
  sd = np.std(input_data, axis=0)
  # Calcular a escala de Pareto
  escala_pareto = 1 / np.sqrt(sd)
  X_pareto = pd.DataFrame(input_data*escala_pareto)
  X_pareto[np.isnan(X_pareto)] = 0

  if mc:
        mean_pareto = np.mean(X_pareto, axis=0)
        X_pareto_mc = X_pareto - mean_pareto
        X_pareto_mc[np.isnan(X_pareto_mc)] = 0
        return pd.DataFrame(X_pareto_mc), mean_pareto
  else:
        return X_pareto

def mc(input_data):
    """
    Mean center
    ----------
    Retorna o dado preprocessado e a media no espaço original
    """
    mean_original = np.mean(input_data, axis=0)
    X_mc = input_data - mean_original
    X_mc[np.isnan(X_mc)] = 0
    return pd.DataFrame(X_mc), mean_original

def auto_scaling(input_data):
    """
    Auto scaling
    ----------
    Retorna o dado preprocessado, sd no espaço original e media no espaço original
    """
    mean_original = np.mean(input_data, axis=0)
    sd_original = np.std(input_data, axis=0)
    X_sc = input_data / sd_original
    X_sc = X_sc - mean_original
    X_sc[np.isnan(X_sc)] = 0
    return pd.DataFrame(X_sc), sd_original, mean_original


def msc(input_data, reference=None):
    import numpy as np
    import pandas as pd
    """
        :msc: Scatter Correction technique performed with mean of the sample data as the reference.
        :param input_data: Array of spectral data
        :type input_data: DataFrame
        :returns: data_msc (ndarray): Scatter corrected spectra data
    """
    eps = np.finfo(np.float32).eps
    input_data = np.array(input_data, dtype=np.float64)
    ref = []
    sampleCount = int(len(input_data))

    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
    
    # Get the reference spectrum. If not given, estimate it from the mean
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        for j in range(0, sampleCount, 10):
            ref.append(np.mean(input_data[j:j+10], axis=0))
            # Run regression
            fit = np.polyfit(ref[i], input_data[i,:], 1, full=True)
            # Apply correction
            data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]
    
    return pd.DataFrame(data_msc)