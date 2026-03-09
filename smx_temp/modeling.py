def vip_scores(pls_model):
    """
    Calculate VIP (Variable Importance in Projection) scores for a fitted PLS model.
    Parameters
    ----------
    - **pls_model** : fitted PLS model object from sklearn.cross_decomposition.PLSRegression
        The PLS model for which to calculate VIP scores.
    Returns
    -------
    - vip_scores : ndarray, shape (n_features,)
        VIP scores for each feature in the model.
    """
    import numpy as np
    import pandas as pd
    
    t = pls_model.x_scores_ # X scores 
    w = pls_model.x_weights_ # X weights
    p = pls_model.y_loadings_ # Y loadings
    features, _ = w.shape # number of features
    vip = np.zeros(shape=(features,)) # initializing VIP scores array
    inner_sum = np.diag(t.T @ t @ p.T @ p) # inner sum calculation
    SS_total = np.sum(inner_sum) # total sum of squares
    vip = np.sqrt(features*(w**2 @ inner_sum)/ SS_total) # VIP calculation
    return pd.DataFrame(vip)

def explained_variance_from_scores(X, T, P, Q=None, Y=None):
    """
    Calculate percent variance explained (based on PCTVAR Matlab function) for X and Y
    by using the scores T and loadings P (and optionally Q for Y).
    Parameters
    ----------
    - **X** : array-like, shape (n_samples, n_features)
        X matrix used in PLS.
    - **T** : array-like, shape (n_samples, n_components)
        Scores matrix from PLS.
    - **P** : array-like, shape (n_features, n_components)
        Loadings matrix for X from PLS.
    - **Q** : array-like, shape (n_targets, n_components), optional
        Loadings matrix for Y from PLS. Required if Yc is provided.
    - **Y** : array-like, shape (n_samples, n_targets), optional
       Y matrix used in PLS.
    Returns
    -------
    - result : dict with keys:
        - **'varX_cumulative'** : ndarray shape (n_components,)
            Percent cumulative variance of X explained by 1..j components.
        - **'varX_per_component'** : ndarray shape (n_components,)
            Percent variance of X explained per component.
        - **'varY_cumulative'** : ndarray shape (n_components,), or None
            Percent cumulative variance of Y explained by 1..j components (if Yc and Q provided).
        - **'varY_per_component'** : ndarray shape (n_components,), or None
            Percent variance of Y explained per component (if Yc and Q provided).
    """
    import numpy as np
    X = np.asarray(X, dtype=float) # X preprocessed data
    T = np.asarray(T, dtype=float) # scores
    P = np.asarray(P, dtype=float) # loadings for X

    n_comp = T.shape[1]
    TSS_X = np.sum(X ** 2) # total sum of squares of X
    if TSS_X == 0: # avoid division by zero
        raise ValueError("TSS_X == 0 (X does not have variability).")

    pctvarX_cum = np.zeros(n_comp, dtype=float) # cumulative percent variance for X

    for j in range(1, n_comp + 1): # loop over components
        Xhat_j = T[:, :j] @ P[:, :j].T # reconstructed X using j components
        SS_Xhat_j = np.sum(Xhat_j ** 2) # sum of squares of reconstructed X
        pctvarX_cum[j-1] = 100.0 * SS_Xhat_j / TSS_X # percent variance explained cumulativa
    
    # incremental (per component)
    pctvarX_per = np.empty_like(pctvarX_cum) # incremental percent variance for X
    pctvarX_per[0] = pctvarX_cum[0] # first component
    pctvarX_per[1:] = pctvarX_cum[1:] - pctvarX_cum[:-1] # rest

    # Y (if provided)
    pctvarY_cum = None # cumulative percent variance for Y
    pctvarY_per = None # incremental percent variance for Y
    if Q is not None and Y is not None: # if Y loadings and Y centered provided
        Q = np.asarray(Q, dtype=float) # loadings for Y
        Y = np.asarray(Y, dtype=float) # centered (and possibly scaled) Y
        TSS_Y = np.sum(Y ** 2) # total sum of squares of Y
        if TSS_Y == 0: # avoid division by zero
            pctvarY_cum = np.zeros(n_comp, dtype=float) # all zeros if Y has no variance
            pctvarY_per = np.zeros(n_comp, dtype=float) # all zeros
        else:
            pctvarY_cum = np.zeros(n_comp, dtype=float) # cumulative percent variance for Y
            for j in range(1, n_comp + 1): # loop over components
                Yhat_j = T[:, :j] @ Q[:, :j].T # reconstructed Y using j components
                SS_Yhat_j = np.sum(Yhat_j ** 2) # sum of squares of reconstructed Y
                pctvarY_cum[j-1] = 100.0 * SS_Yhat_j / TSS_Y # percent variance explained cumulativa
            pctvarY_per = np.empty_like(pctvarY_cum) # incremental percent variance for Y
            pctvarY_per[0] = pctvarY_cum[0] # first component
            pctvarY_per[1:] = pctvarY_cum[1:] - pctvarY_cum[:-1] # rest

        return {
            'varX_cumulative': pctvarX_cum[-1],
            'varX_per_component': pctvarX_per[-1],
            'varY_cumulative': pctvarY_cum[-1],
            'varY_per_component': pctvarY_per[-1]
            }         


def pls_optimized(Xcal, ycal, LVmax, Xpred=None, ypred=None, aim='regression', cv=10):
    """
    ## PLS optimized
    Function to fit a PLS regression or PLS-DA model with optimization of latent variables (LVs)
    using cross-validation. It calculates various performance metrics for calibration, cross-validation,
    and prediction (if provided) datasets
    **Parameters**:
    - **Xcal** : pd.DataFrame
        Calibration dataset features.
    - **ycal** : pd.Series or np.ndarray
        Calibration dataset target variable (regression) or binary class labels (classification).
    - **LVmax** : int
        Maximum number of latent variables to consider.
    - **Xpred** : pd.DataFrame, optional
        Prediction dataset features. Default is None.
    - **ypred** : pd.Series or np.ndarray, optional
        Prediction dataset target variable (regression) or binary class labels (classification). Default is None.
    - **aim** : str, optional
        Type of analysis: 'regression' for PLS regression or 'classification' for PLS-DA. Default is 'regression'.
    - **cv** : int, optional
        Number of cross-validation folds. Default is 10
        
    **Returns**:
    - **df_results** : pd.DataFrame
        DataFrame containing performance metrics for each number of latent variables.
    - **calres** : pd.DataFrame
        DataFrame containing predicted values for the calibration dataset.
    - **predres** : pd.DataFrame
        DataFrame containing predicted values for the prediction dataset (if provided).
    """

    import numpy as np
    import pandas as pd

    if aim == 'regression': # regression (PLSR)
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import mean_squared_error, r2_score
        from scipy.stats import iqr
        
        results = [] # list to store results for each LV
        calres = pd.DataFrame(index=range(len(ycal))) # calibration results
        predres = pd.DataFrame(index=range(len(ypred))) if (Xpred is not None and ypred is not None) else None # prediction results

        for n_comp in range(1, LVmax + 1): # loop over number of components
            plsr = PLSRegression(n_components=n_comp, scale=False)
            plsr.fit(Xcal, ycal)
            y_cal = plsr.predict(Xcal).flatten()
            calres[f'LV_{n_comp}'] = y_cal

            y_cv = cross_val_predict(plsr, Xcal, ycal, cv=cv) # cross-validated predictions
            y_cv = np.array(y_cv).flatten()

            R2_cal = r2_score(ycal, y_cal) # determination coefficient
            r2_cal = np.corrcoef(ycal, y_cal)[0, 1] ** 2 # correlation coefficient squared
            rmse_cal = np.sqrt(mean_squared_error(ycal, y_cal))
            R2_cv = r2_score(ycal, y_cv)
            r2_cv = np.corrcoef(ycal, y_cv)[0, 1] ** 2
            rmsecv = np.sqrt(mean_squared_error(ycal, y_cv))
            rpd_cv = ycal.std() / rmsecv if rmsecv != 0 else np.nan
            rpiq_cv = iqr(ycal, rng=(25, 75)) / rmsecv if rmsecv != 0 else np.nan
            bias_cv = np.sum(ycal - y_cv) / ycal.shape[0]
            SDV_cv = (ycal - y_cv) - bias_cv
            SDV_cv = np.sqrt(np.sum(SDV_cv * SDV_cv) / (ycal.shape[0] - 1)) if ycal.shape[0] > 1 else np.nan
            tbias_cv = abs(bias_cv) * (np.sqrt(ycal.shape[0]) / SDV_cv) if SDV_cv not in (0, np.nan) else np.nan
            
            # explained variance
            exp_var = explained_variance_from_scores(Xcal, plsr.x_scores_, plsr.x_loadings_,
                                               Q=plsr.y_loadings_, Y=ycal) # explained variance
            
            # vip scores
            vip = vip_scores(plsr).T
            vip.columns = plsr.feature_names_in_ # setting feature names

            if Xpred is not None and ypred is not None: # prediction set
                y_pred = plsr.predict(Xpred).flatten()
                predres[f'LV_{n_comp}'] = y_pred

                R2_pred = r2_score(ypred, y_pred) # determination coefficient
                r2_pred = np.corrcoef(ypred, y_pred)[0, 1] ** 2 # correlation coefficient squared
                rmsep = np.sqrt(mean_squared_error(ypred, y_pred))
                rpd_pred = ypred.std() / rmsep if rmsep != 0 else np.nan
                rpiq_pred = iqr(ypred, rng=(25, 75)) / rmsep if rmsep != 0 else np.nan
                bias_pred = np.sum(ypred - y_pred) / ypred.shape[0]
                SDV_pred = (ypred - y_pred) - bias_pred
                SDV_pred = np.sqrt(np.sum(SDV_pred * SDV_pred) / (ypred.shape[0] - 1)) if ypred.shape[0] > 1 else np.nan
                tbias_pred = abs(bias_pred) * (np.sqrt(ypred.shape[0]) / SDV_pred) if SDV_pred not in (0, np.nan) else np.nan
            else:
                r2_pred = rmsep = rpd_pred = rpiq_pred = bias_pred = tbias_pred = None

            results.append({
                'LVs': n_comp,
                'R2_Cal': R2_cal,
                'r2_Cal': r2_cal,
                'RMSEC': rmse_cal,
                'R2_CV': R2_cv,
                'r2_Cv': r2_cv,
                'RMSECV': rmsecv,
                'RPD_CV': rpd_cv,
                'RPIQ_CV': rpiq_cv,
                'Bias_CV': bias_cv,
                'tbias_CV': tbias_cv,
                'R2_Pred': R2_pred,
                'r2_Pred': r2_pred,
                'RMSEP': rmsep,
                'RPD_Pred': rpd_pred,
                'RPIQ_Pred': rpiq_pred,
                'Bias_Pred': bias_pred,
                'tbias_Pred': tbias_pred,
                'X_Cum_Exp_Var' : exp_var['varX_cumulative'],
                'Y_Cum_Exp_Var' : exp_var['varY_cumulative'],
                'X_Ind_Exp_Var' : exp_var['varX_per_component'],
                'Y_Ind_Exp_Var' : exp_var['varY_per_component']
            })

        model = plsr  # last model fitted
        df_results = pd.DataFrame(results)
        calres.insert(0, 'Ref', np.array(ycal))
        if predres is not None:
            predres.insert(0, 'Ref', np.array(ypred))    

    elif aim == 'classification': # classification (PLS-DA)
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import accuracy_score, confusion_matrix

        results = []
        calres = pd.DataFrame(index=range(len(ycal))) # calibration results
        predres = pd.DataFrame(index=range(len(ypred))) if (Xpred is not None and ypred is not None) else None # prediction results

        # ensure binary classes
        ycal_series = pd.Series(ycal).reset_index(drop=True) # ensure it's a Series
        unique_labels = ycal_series.unique() # unique class labels
        if len(unique_labels) != 2: # check for binary classification
            raise ValueError(f"PLS-DA (this function) expects 2 classes (binary). Found: {unique_labels}")

        label_to_num = {lab: idx for idx, lab in enumerate(unique_labels)} # mapping labels to 0 and 1
        num_to_label = {idx: lab for lab, idx in label_to_num.items()} # reverse mapping for predictions
       
        # prepare ycal numeric
        ycal_numeric = np.array([label_to_num[i] for i in ycal]) 

        # prepare ypred numeric if provided
        ypred_numeric = None
        if ypred is not None:
            ypred_numeric = np.array([label_to_num[i] for i in ypred])

        for n_comp in range(1, LVmax + 1): # loop over number of components
            plsda = PLSRegression(n_components=n_comp, scale=False)
            plsda.fit(Xcal, ycal_numeric)

            # calibration continuous predictions -> binarize
            y_cal_cont = plsda.predict(Xcal).flatten()
            y_cal_bin = (y_cal_cont >= 0.5).astype(int)
            y_cal_class = np.array([num_to_label[i] for i in y_cal_bin])
            calres[f'LV_{n_comp}'] = y_cal_class
            calres_numeric = pd.DataFrame(y_cal_cont, columns=[f'LV_{n_comp}']) # numeric calibration results

            # cross-validated continuous predictions -> binarize
            y_cv_cont = cross_val_predict(plsda, Xcal, ycal_numeric, cv=cv)
            y_cv_cont = np.array(y_cv_cont).flatten()
            y_cv_bin = (y_cv_cont >= 0.5).astype(int)

            # metrics
            acc_cal = accuracy_score(ycal_numeric, y_cal_bin)
            cm_cal = confusion_matrix(ycal_numeric, y_cal_bin)
            # safe unpack for binary confusion matrix
            if cm_cal.size == 4:
                tn, fp, fn, tp = cm_cal.ravel()
            else:
                tn = fp = fn = tp = np.nan
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

            acc_cv = accuracy_score(ycal_numeric, y_cv_bin)
            cm_cv = confusion_matrix(ycal_numeric, y_cv_bin)
            if cm_cv.size == 4:
                tn_cv, fp_cv, fn_cv, tp_cv = cm_cv.ravel()
            else:
                tn_cv = fp_cv = fn_cv = tp_cv = np.nan
            sensitivity_cv = tp_cv / (tp_cv + fn_cv) if (tp_cv + fn_cv) > 0 else np.nan
            specificity_cv = tn_cv / (tn_cv + fp_cv) if (tn_cv + fp_cv) > 0 else np.nan

            # explained variance
            exp_var = explained_variance_from_scores(Xcal, plsda.x_scores_, plsda.x_loadings_,
                                               Q=plsda.y_loadings_, Y=ycal_numeric.reshape(-1, 1)) # explained variance

            # vip scores
            vip = vip_scores(plsda).T
            vip.columns = plsda.feature_names_in_ # setting feature names

            # prediction set (if provided)
            if Xpred is not None and ypred is not None:
                y_pred_cont = plsda.predict(Xpred).flatten()
                y_pred_bin = (y_pred_cont >= 0.5).astype(int)
                y_pred_class = np.array([num_to_label[i] for i in y_pred_bin])
                predres[f'LV_{n_comp}'] = y_pred_class
                predres_numeric = pd.DataFrame(y_pred_cont, columns=[f'LV_{n_comp}']) # numeric prediction results

                acc_pred = accuracy_score(ypred_numeric, y_pred_bin)
                cm_pred = confusion_matrix(ypred_numeric, y_pred_bin)
                if cm_pred.size == 4:
                    tn_p, fp_p, fn_p, tp_p = cm_pred.ravel()
                else:
                    tn_p = fp_p = fn_p = tp_p = np.nan
                sensitivity_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else np.nan
                specificity_p = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else np.nan
            else:
                acc_pred = sensitivity_p = specificity_p = cm_pred = tn_p = fp_p = fn_p = tp_p = None

            results.append({
                'LVs': n_comp,
                'Accuracy Cal': acc_cal,
                'Sensitivity Cal': sensitivity,
                'Specificity Cal': specificity,
                'CM Cal': cm_cal,
                'Accuracy CV': acc_cv,
                'Sensitivity CV': sensitivity_cv,
                'Specificity CV': specificity_cv,
                'CM CV': cm_cv,
                'Accuracy Pred': acc_pred,
                'Sensitivity Pred': sensitivity_p,
                'Specificity Pred': specificity_p,
                'CM Pred': cm_pred,
                'X Cum Exp Var' : exp_var['varX_cumulative'],
                'Y Cum Exp Var' : exp_var['varY_cumulative'],
                'X Ind Exp Var' : exp_var['varX_per_component'],
                'Y Ind Exp Var' : exp_var['varY_per_component']
            })

        model = plsda  # last model fitted
        df_results = pd.DataFrame(results)
        calres.insert(0, 'Ref', np.array(ycal))
        if predres is not None:
            predres.insert(0, 'Ref', np.array(ypred))

    else:
        raise ValueError("Parameter `aim` must be 'regression' or 'classification'.")

    if aim == 'classification':
        return df_results, calres, predres, model, vip, calres_numeric, predres_numeric
    else:
        return df_results, calres, predres, model, vip


def svm_optimized(Xcal, ycal, Xpred=None, ypred=None, aim='regression', **svm_params):
    """
    ## SVM optimized
    Function to fit an SVM regression (SVR) or SVM classification (SVC) model.
    It calculates various performance metrics for calibration and prediction (if provided) datasets.
    
    **Parameters**:
    - **Xcal** : pd.DataFrame
        Calibration dataset features.
    - **ycal** : pd.Series or np.ndarray
        Calibration dataset target variable (regression) or class labels (classification).
    - **Xpred** : pd.DataFrame, optional
        Prediction dataset features. Default is None.
    - **ypred** : pd.Series or np.ndarray, optional
        Prediction dataset target variable (regression) or class labels (classification). Default is None.
    - **aim** : str, optional
        Type of analysis: 'regression' for SVR or 'classification' for SVC. Default is 'regression'.
    - **svm_params** : dict, optional
        Additional hyperparameters for SVM (e.g., C, kernel, gamma, epsilon for SVR, etc.).
        If not provided, sklearn defaults will be used.
        
    **Returns**:
    - **df_results** : pd.DataFrame
        DataFrame containing performance metrics.
    - **calres** : pd.DataFrame
        DataFrame containing predicted values for the calibration dataset.
    - **predres** : pd.DataFrame
        DataFrame containing predicted values for the prediction dataset (if provided).
    - **model** : fitted SVM model
        The fitted SVR or SVC model.
    
    For classification (aim='classification'), additional returns:
    - **calres_proba** : pd.DataFrame
        DataFrame with predict_proba outputs for calibration (probability of positive class).
    - **predres_proba** : pd.DataFrame
        DataFrame with predict_proba outputs for prediction (if provided).
    - **calres_decision** : pd.DataFrame
        DataFrame with decision_function outputs for calibration (distance to hyperplane).
    - **predres_decision** : pd.DataFrame
        DataFrame with decision_function outputs for prediction (if provided).
    """

    import numpy as np
    import pandas as pd

    if aim == 'regression':  # regression (SVR)
        from sklearn.svm import SVR
        from sklearn.metrics import mean_squared_error, r2_score
        from scipy.stats import iqr
        
        results = []  # list to store results
        calres = pd.DataFrame(index=range(len(ycal)))  # calibration results
        predres = pd.DataFrame(index=range(len(ypred))) if (Xpred is not None and ypred is not None) else None

        # fit SVR model
        svr = SVR(**svm_params)
        svr.fit(Xcal, ycal)
        y_cal = svr.predict(Xcal).flatten()
        calres['SVR'] = y_cal

        # calibration metrics
        R2_cal = r2_score(ycal, y_cal)
        r2_cal = np.corrcoef(ycal, y_cal)[0, 1] ** 2
        rmse_cal = np.sqrt(mean_squared_error(ycal, y_cal))

        # prediction set metrics (if provided)
        if Xpred is not None and ypred is not None:
            y_pred = svr.predict(Xpred).flatten()
            predres['SVR'] = y_pred

            R2_pred = r2_score(ypred, y_pred)
            r2_pred = np.corrcoef(ypred, y_pred)[0, 1] ** 2
            rmsep = np.sqrt(mean_squared_error(ypred, y_pred))
            rpd_pred = ypred.std() / rmsep if rmsep != 0 else np.nan
            rpiq_pred = iqr(ypred, rng=(25, 75)) / rmsep if rmsep != 0 else np.nan
            bias_pred = np.sum(ypred - y_pred) / ypred.shape[0]
            SDV_pred = (ypred - y_pred) - bias_pred
            SDV_pred = np.sqrt(np.sum(SDV_pred * SDV_pred) / (ypred.shape[0] - 1)) if ypred.shape[0] > 1 else np.nan
            tbias_pred = abs(bias_pred) * (np.sqrt(ypred.shape[0]) / SDV_pred) if SDV_pred not in (0, np.nan) else np.nan
        else:
            R2_pred = r2_pred = rmsep = rpd_pred = rpiq_pred = bias_pred = tbias_pred = None

        results.append({
            'Model': 'SVR',
            'R2_Cal': R2_cal,
            'r2_Cal': r2_cal,
            'RMSEC': rmse_cal,
            'R2_Pred': R2_pred,
            'r2_Pred': r2_pred,
            'RMSEP': rmsep,
            'RPD_Pred': rpd_pred,
            'RPIQ_Pred': rpiq_pred,
            'Bias_Pred': bias_pred,
            'tbias_Pred': tbias_pred
        })

        model = svr
        df_results = pd.DataFrame(results)
        calres.insert(0, 'Ref', np.array(ycal))
        if predres is not None:
            predres.insert(0, 'Ref', np.array(ypred))

        return df_results, calres, predres, model

    elif aim == 'classification':  # classification (SVC)
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, confusion_matrix

        results = []
        calres = pd.DataFrame(index=range(len(ycal)))  # calibration results (class labels)
        predres = pd.DataFrame(index=range(len(ypred))) if (Xpred is not None and ypred is not None) else None

        # DataFrames for numeric outputs
        calres_proba = pd.DataFrame(index=range(len(ycal)))  # predict_proba outputs
        calres_decision = pd.DataFrame(index=range(len(ycal)))  # decision_function outputs
        predres_proba = pd.DataFrame(index=range(len(ypred))) if (Xpred is not None and ypred is not None) else None
        predres_decision = pd.DataFrame(index=range(len(ypred))) if (Xpred is not None and ypred is not None) else None

        # ensure binary classes
        ycal_series = pd.Series(ycal).reset_index(drop=True)
        unique_labels = ycal_series.unique()
        if len(unique_labels) != 2:
            raise ValueError(f"SVC (this function) expects 2 classes (binary). Found: {unique_labels}")

        label_to_num = {lab: idx for idx, lab in enumerate(unique_labels)}
        num_to_label = {idx: lab for lab, idx in label_to_num.items()}
       
        # prepare ycal numeric
        ycal_numeric = np.array([label_to_num[i] for i in ycal])

        # prepare ypred numeric if provided
        ypred_numeric = None
        if ypred is not None:
            ypred_numeric = np.array([label_to_num[i] for i in ypred])

        # ensure probability=True for predict_proba
        svm_params_with_proba = svm_params.copy()
        svm_params_with_proba['probability'] = True

        # fit SVC model
        svc = SVC(**svm_params_with_proba)
        svc.fit(Xcal, ycal_numeric)

        # calibration predictions
        y_cal_pred = svc.predict(Xcal)
        y_cal_class = np.array([num_to_label[i] for i in y_cal_pred])
        calres['SVC'] = y_cal_class

        # calibration numeric outputs
        y_cal_proba = svc.predict_proba(Xcal)[:, 1]  # probability of positive class
        y_cal_decision = svc.decision_function(Xcal)  # distance to hyperplane
        calres_proba['SVC'] = y_cal_proba
        calres_decision['SVC'] = y_cal_decision

        # calibration metrics
        acc_cal = accuracy_score(ycal_numeric, y_cal_pred)
        cm_cal = confusion_matrix(ycal_numeric, y_cal_pred)
        if cm_cal.size == 4:
            tn, fp, fn, tp = cm_cal.ravel()
        else:
            tn = fp = fn = tp = np.nan
        sensitivity_cal = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity_cal = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        # prediction set (if provided)
        if Xpred is not None and ypred is not None:
            y_pred_pred = svc.predict(Xpred)
            y_pred_class = np.array([num_to_label[i] for i in y_pred_pred])
            predres['SVC'] = y_pred_class

            # prediction numeric outputs
            y_pred_proba = svc.predict_proba(Xpred)[:, 1]
            y_pred_decision = svc.decision_function(Xpred)
            predres_proba['SVC'] = y_pred_proba
            predres_decision['SVC'] = y_pred_decision

            acc_pred = accuracy_score(ypred_numeric, y_pred_pred)
            cm_pred = confusion_matrix(ypred_numeric, y_pred_pred)
            if cm_pred.size == 4:
                tn_p, fp_p, fn_p, tp_p = cm_pred.ravel()
            else:
                tn_p = fp_p = fn_p = tp_p = np.nan
            sensitivity_pred = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else np.nan
            specificity_pred = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else np.nan
        else:
            acc_pred = sensitivity_pred = specificity_pred = cm_pred = None

        results.append({
            'Model': 'SVC',
            'Accuracy Cal': acc_cal,
            'Sensitivity Cal': sensitivity_cal,
            'Specificity Cal': specificity_cal,
            'CM Cal': cm_cal,
            'Accuracy Pred': acc_pred,
            'Sensitivity Pred': sensitivity_pred,
            'Specificity Pred': specificity_pred,
            'CM Pred': cm_pred
        })

        model = svc
        df_results = pd.DataFrame(results)
        calres.insert(0, 'Ref', np.array(ycal))
        if predres is not None:
            predres.insert(0, 'Ref', np.array(ypred))

        return df_results, calres, predres, model, calres_proba, predres_proba, calres_decision, predres_decision

    else:
        raise ValueError("Parameter `aim` must be 'regression' or 'classification'.")


def mlp_optimized(Xcal, ycal, Xpred=None, ypred=None, aim='regression', **mlp_params):
    """
    ## MLP optimized
    Function to fit an MLP regression (MLPRegressor) or MLP classification (MLPClassifier) model.
    It calculates various performance metrics for calibration and prediction (if provided) datasets.
    
    **Parameters**:
    - **Xcal** : pd.DataFrame
        Calibration dataset features.
    - **ycal** : pd.Series or np.ndarray
        Calibration dataset target variable (regression) or class labels (classification).
    - **Xpred** : pd.DataFrame, optional
        Prediction dataset features. Default is None.
    - **ypred** : pd.Series or np.ndarray, optional
        Prediction dataset target variable (regression) or class labels (classification). Default is None.
    - **aim** : str, optional
        Type of analysis: 'regression' for MLPRegressor or 'classification' for MLPClassifier. Default is 'regression'.
    - **mlp_params** : dict, optional
        Additional hyperparameters for MLP (e.g., hidden_layer_sizes, activation, solver, alpha, max_iter, etc.).
        If not provided, sklearn defaults will be used.
        
    **Returns**:
    - **df_results** : pd.DataFrame
        DataFrame containing performance metrics.
    - **calres** : pd.DataFrame
        DataFrame containing predicted values for the calibration dataset.
    - **predres** : pd.DataFrame
        DataFrame containing predicted values for the prediction dataset (if provided).
    - **model** : fitted MLP model
        The fitted MLPRegressor or MLPClassifier model.
    
    For classification (aim='classification'), additional returns:
    - **calres_proba** : pd.DataFrame
        DataFrame with predict_proba outputs for calibration (probability of positive class).
    - **predres_proba** : pd.DataFrame
        DataFrame with predict_proba outputs for prediction (if provided).
    """

    import numpy as np
    import pandas as pd

    if aim == 'regression':  # regression (MLPRegressor)
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from scipy.stats import iqr
        
        results = []  # list to store results
        calres = pd.DataFrame(index=range(len(ycal)))  # calibration results
        predres = pd.DataFrame(index=range(len(ypred))) if (Xpred is not None and ypred is not None) else None

        # fit MLPRegressor model
        mlp = MLPRegressor(**mlp_params)
        mlp.fit(Xcal, ycal)
        y_cal = mlp.predict(Xcal).flatten()
        calres['MLP'] = y_cal

        # calibration metrics
        R2_cal = r2_score(ycal, y_cal)
        r2_cal = np.corrcoef(ycal, y_cal)[0, 1] ** 2
        rmse_cal = np.sqrt(mean_squared_error(ycal, y_cal))

        # prediction set metrics (if provided)
        if Xpred is not None and ypred is not None:
            y_pred = mlp.predict(Xpred).flatten()
            predres['MLP'] = y_pred

            R2_pred = r2_score(ypred, y_pred)
            r2_pred = np.corrcoef(ypred, y_pred)[0, 1] ** 2
            rmsep = np.sqrt(mean_squared_error(ypred, y_pred))
            rpd_pred = ypred.std() / rmsep if rmsep != 0 else np.nan
            rpiq_pred = iqr(ypred, rng=(25, 75)) / rmsep if rmsep != 0 else np.nan
            bias_pred = np.sum(ypred - y_pred) / ypred.shape[0]
            SDV_pred = (ypred - y_pred) - bias_pred
            SDV_pred = np.sqrt(np.sum(SDV_pred * SDV_pred) / (ypred.shape[0] - 1)) if ypred.shape[0] > 1 else np.nan
            tbias_pred = abs(bias_pred) * (np.sqrt(ypred.shape[0]) / SDV_pred) if SDV_pred not in (0, np.nan) else np.nan
        else:
            R2_pred = r2_pred = rmsep = rpd_pred = rpiq_pred = bias_pred = tbias_pred = None

        results.append({
            'Model': 'MLP',
            'R2_Cal': R2_cal,
            'r2_Cal': r2_cal,
            'RMSEC': rmse_cal,
            'R2_Pred': R2_pred,
            'r2_Pred': r2_pred,
            'RMSEP': rmsep,
            'RPD_Pred': rpd_pred,
            'RPIQ_Pred': rpiq_pred,
            'Bias_Pred': bias_pred,
            'tbias_Pred': tbias_pred
        })

        model = mlp
        df_results = pd.DataFrame(results)
        calres.insert(0, 'Ref', np.array(ycal))
        if predres is not None:
            predres.insert(0, 'Ref', np.array(ypred))

        return df_results, calres, predres, model

    elif aim == 'classification':  # classification (MLPClassifier)
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix

        results = []
        calres = pd.DataFrame(index=range(len(ycal)))  # calibration results (class labels)
        predres = pd.DataFrame(index=range(len(ypred))) if (Xpred is not None and ypred is not None) else None

        # DataFrames for numeric outputs
        calres_proba = pd.DataFrame(index=range(len(ycal)))  # predict_proba outputs
        predres_proba = pd.DataFrame(index=range(len(ypred))) if (Xpred is not None and ypred is not None) else None

        # ensure binary classes
        ycal_series = pd.Series(ycal).reset_index(drop=True)
        unique_labels = ycal_series.unique()
        if len(unique_labels) != 2:
            raise ValueError(f"MLPClassifier (this function) expects 2 classes (binary). Found: {unique_labels}")

        label_to_num = {lab: idx for idx, lab in enumerate(unique_labels)}
        num_to_label = {idx: lab for lab, idx in label_to_num.items()}
       
        # prepare ycal numeric
        ycal_numeric = np.array([label_to_num[i] for i in ycal])

        # prepare ypred numeric if provided
        ypred_numeric = None
        if ypred is not None:
            ypred_numeric = np.array([label_to_num[i] for i in ypred])

        # fit MLPClassifier model
        mlp = MLPClassifier(**mlp_params)
        mlp.fit(Xcal, ycal_numeric)

        # calibration predictions
        y_cal_pred = mlp.predict(Xcal)
        y_cal_class = np.array([num_to_label[i] for i in y_cal_pred])
        calres['MLP'] = y_cal_class

        # calibration numeric outputs
        y_cal_proba = mlp.predict_proba(Xcal)[:, 1]  # probability of positive class
        calres_proba['MLP'] = y_cal_proba

        # calibration metrics
        acc_cal = accuracy_score(ycal_numeric, y_cal_pred)
        cm_cal = confusion_matrix(ycal_numeric, y_cal_pred)
        if cm_cal.size == 4:
            tn, fp, fn, tp = cm_cal.ravel()
        else:
            tn = fp = fn = tp = np.nan
        sensitivity_cal = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity_cal = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        # prediction set (if provided)
        if Xpred is not None and ypred is not None:
            y_pred_pred = mlp.predict(Xpred)
            y_pred_class = np.array([num_to_label[i] for i in y_pred_pred])
            predres['MLP'] = y_pred_class

            # prediction numeric outputs
            y_pred_proba = mlp.predict_proba(Xpred)[:, 1]
            predres_proba['MLP'] = y_pred_proba

            acc_pred = accuracy_score(ypred_numeric, y_pred_pred)
            cm_pred = confusion_matrix(ypred_numeric, y_pred_pred)
            if cm_pred.size == 4:
                tn_p, fp_p, fn_p, tp_p = cm_pred.ravel()
            else:
                tn_p = fp_p = fn_p = tp_p = np.nan
            sensitivity_pred = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else np.nan
            specificity_pred = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else np.nan
        else:
            acc_pred = sensitivity_pred = specificity_pred = cm_pred = None

        results.append({
            'Model': 'MLP',
            'Accuracy Cal': acc_cal,
            'Sensitivity Cal': sensitivity_cal,
            'Specificity Cal': specificity_cal,
            'CM Cal': cm_cal,
            'Accuracy Pred': acc_pred,
            'Sensitivity Pred': sensitivity_pred,
            'Specificity Pred': specificity_pred,
            'CM Pred': cm_pred
        })

        model = mlp
        df_results = pd.DataFrame(results)
        calres.insert(0, 'Ref', np.array(ycal))
        if predres is not None:
            predres.insert(0, 'Ref', np.array(ypred))

        return df_results, calres, predres, model, calres_proba, predres_proba

    else:
        raise ValueError("Parameter `aim` must be 'regression' or 'classification'.")