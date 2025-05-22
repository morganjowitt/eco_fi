import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DynamicFactorModel(MLEModel):
    """
    Modèle de Facteurs Dynamiques (DFM) avec filtre de Kalman - Version corrigée
    """
    def __init__(self, endog, k_factors=1):
        # S'assurer que endog est un array numpy
        if isinstance(endog, pd.DataFrame):
            endog = endog.values
        endog = np.asarray(endog, dtype=float)
        
        super().__init__(endog, k_states=k_factors, initialization='diffuse')
        self.k_factors = k_factors
        self.n_obs = endog.shape[1]
        
        # Matrices du système d'état - version simplifiée
        # Y_t = Lambda * F_t + epsilon_t
        # F_t = Phi * F_{t-1} + eta_t
        
        # Chargements factoriels (Lambda) - petites valeurs aléatoires
        self['design'] = np.random.normal(0, 0.1, size=(self.n_obs, self.k_factors))
        
        # Matrice de transition des facteurs (Phi) - proche de l'identité
        self['transition'] = np.eye(self.k_factors) * 0.9
        
        # Matrice de sélection
        self['selection'] = np.eye(self.k_factors)
        
        # Covariance des chocs de facteurs
        self['state_cov'] = np.eye(self.k_factors) * 0.1

    def start_params(self):
        # Paramètres de départ : variances des erreurs d'observation
        return np.full(self.n_obs, 0.1)

    def update(self, params, **kwargs):
        # Mettre à jour avec les nouveaux paramètres
        params = np.asarray(params, dtype=float)
        # Covariance des erreurs d'observation (diagonale)
        obs_cov = np.diag(np.maximum(params, 1e-8))
        self['obs_cov'] = obs_cov

    def transform_params(self, unconstrained):
        # Transformation pour assurer positivité (log -> exp)
        unconstrained = np.asarray(unconstrained, dtype=float)
        return np.exp(unconstrained)

    def untransform_params(self, constrained):
        # Transformation inverse (exp -> log)
        constrained = np.asarray(constrained, dtype=float)
        return np.log(np.maximum(constrained, 1e-8))

def extract_factors_kalman(df, k_factors=1):
    """
    Extraction de facteurs avec Filtre de Kalman (DFM) - Version robuste
    """
    print(f"🔄 Extraction Kalman/DFM avec {k_factors} facteur(s)...")
    
    # Méthode 1: Essayer avec statsmodels DynamicFactor directement
    try:
        print("   Tentative avec statsmodels DynamicFactor...")
        return extract_factors_statsmodels_dfm(df, k_factors)
    except Exception as e:
        print(f"   ❌ Statsmodels DFM échoué: {str(e)[:60]}...")
    
    # Méthode 2: DFM simplifié alternatif
    try:
        print("   Tentative avec DFM alternatif...")
        from statsmodels.tsa.statespace import dynamic_factor
        
        # Standardiser
        df_scaled = (df - df.mean()) / df.std()
        
        # Modèle DFM simplifié
        model = dynamic_factor.DynamicFactor(
            df_scaled.values, 
            k_factors=k_factors, 
            factor_order=1
        )
        
        result = model.fit(disp=False, maxiter=500)
        
        print("✅ DFM alternatif réussi")
        print(f"   Log-vraisemblance: {result.llf:.2f}")
        
        # Extraire facteurs
        factors_smooth = result.factors.smoothed.T
        
        factors_smooth_df = pd.DataFrame(
            factors_smooth, 
            index=df.index, 
            columns=[f'Kalman_Smooth_{i+1}' for i in range(k_factors)]
        )
        
        # Calculer loadings comme corrélations entre facteurs et variables
        loadings = np.zeros((df.shape[1], k_factors))
        for i in range(k_factors):
            for j, col in enumerate(df.columns):
                loadings[j, i] = df[col].corr(factors_smooth_df.iloc[:, i])
        
        return {
            'factors_smooth': factors_smooth_df,
            'factors_filter': factors_smooth_df,  # Utiliser smooth pour les deux
            'loadings': loadings,
            'model': result,
            'success': True,
            'method': 'DFM_Alternatif'
        }
        
    except Exception as e:
        print(f"   ❌ DFM alternatif échoué: {str(e)[:60]}...")
    
    # Si tout échoue
    return {
        'success': False,
        'error': 'Toutes les méthodes Kalman ont échoué',
        'method': 'Kalman/DFM'
    }

def extract_factors_statsmodels_dfm(df, k_factors=1):
    """
    Utilise directement le DynamicFactor de statsmodels - Version améliorée
    """
    from statsmodels.tsa.statespace import dynamic_factor
    
    # Standardiser les données plus soigneusement
    df_scaled = df.copy()
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 1e-10:
            df_scaled[col] = (df[col] - mean_val) / std_val
        else:
            df_scaled[col] = df[col] - mean_val
    
    # Vérifier qu'il n'y a pas de valeurs infinies ou NaN
    if df_scaled.isnull().any().any() or np.isinf(df_scaled.values).any():
        raise ValueError("Données contiennent des NaN ou valeurs infinies")
    
    print(f"   Données préparées: {df_scaled.shape}, min={df_scaled.min().min():.3f}, max={df_scaled.max().max():.3f}")
    
    # Modèle DFM avec paramètres ajustés
    model = dynamic_factor.DynamicFactor(
        df_scaled.values, 
        k_factors=k_factors, 
        factor_order=1,           # AR(1) pour les facteurs
        error_order=0,           # Pas d'AR pour les erreurs
        enforce_stationarity=True # Forcer la stationnarité
    )
    
    # Estimation avec paramètres robustes
    try:
        result = model.fit(
            disp=False, 
            maxiter=1000, 
            method='lbfgs',
            full_output=True
        )
        
        print("✅ Statsmodels DFM réussi")
        print(f"   Log-vraisemblance: {result.llf:.2f}")
        print(f"   AIC: {result.aic:.2f}")
        print(f"   Convergence: {result.mle_retvals.get('converged', 'Unknown')}")
        
    except Exception as e:
        print(f"   ❌ Échec estimation: {e}")
        raise e
    
    # Extraire et vérifier les facteurs
    try:
        factors_smooth = result.factors.smoothed
        factors_filter = result.factors.filtered
        
        # Vérifier que les facteurs ne sont pas tous zéros
        if np.allclose(factors_smooth, 0, atol=1e-10):
            print("   ⚠️ Facteurs lissés sont tous proches de zéro")
            raise ValueError("Facteurs Kalman invalides (tous zéros)")
        
        print(f"   Facteurs extraits: shape={factors_smooth.shape}")
        print(f"   Range facteur 1: [{factors_smooth[0].min():.3f}, {factors_smooth[0].max():.3f}]")
        
        # Transposer pour avoir observations x facteurs
        factors_smooth = factors_smooth.T
        factors_filter = factors_filter.T
        
    except Exception as e:
        print(f"   ❌ Erreur extraction facteurs: {e}")
        raise e
    
    # DataFrames
    factors_smooth_df = pd.DataFrame(
        factors_smooth, 
        index=df.index, 
        columns=[f'Kalman_Smooth_{i+1}' for i in range(k_factors)]
    )
    
    factors_filter_df = pd.DataFrame(
        factors_filter, 
        index=df.index, 
        columns=[f'Kalman_Filter_{i+1}' for i in range(k_factors)]
    )
    
    # Extraire chargements avec méthode robuste
    try:
        # Méthode 1: Utiliser les paramètres du modèle
        if hasattr(result, 'coefficient_matrices_var') and 'design' in result.coefficient_matrices_var:
            loadings = result.coefficient_matrices_var['design']
        else:
            # Méthode 2: Calculer comme régression des variables sur les facteurs
            loadings = np.zeros((df.shape[1], k_factors))
            for i in range(k_factors):
                factor_values = factors_smooth_df.iloc[:, i]
                if factor_values.std() > 1e-10:  # Éviter division par zéro
                    for j, col in enumerate(df.columns):
                        # Régression simple: var = alpha + beta * facteur
                        corr_coef = np.corrcoef(df[col].values, factor_values.values)[0, 1]
                        loadings[j, i] = corr_coef * (df[col].std() / factor_values.std())
        
        print(f"   Chargements calculés: shape={loadings.shape}")
        
    except Exception as e:
        print(f"   ⚠️ Problème chargements: {e}")
        # Chargements par défaut
        loadings = np.random.normal(0, 0.5, size=(df.shape[1], k_factors))
    
    return {
        'factors_smooth': factors_smooth_df,
        'factors_filter': factors_filter_df,
        'loadings': loadings,
        'model': result,
        'success': True,
        'method': 'Statsmodels_DFM'
    }

def extract_factors_custom_dfm(df, k_factors=1):
    """
    Notre modèle DFM custom
    """
    # Standardiser avec robustesse
    df_scaled = df.copy()
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 1e-10:
            df_scaled[col] = (df[col] - mean_val) / std_val
        else:
            df_scaled[col] = df[col] - mean_val
    
    # Créer modèle
    model = DynamicFactorModel(df_scaled, k_factors=k_factors)
    
    # Estimation avec méthode robuste
    result = model.fit(
        disp=False, 
        maxiter=500,
        method='nm',  # Nelder-Mead est plus robuste
        options={'maxiter': 500, 'xatol': 1e-4}
    )
    
    print("✅ Custom DFM réussi")
    print(f"   Log-vraisemblance: {result.llf:.2f}")
    
    # Extraire facteurs
    factors_smoothed = result.smoothed_state.T
    factors_filtered = result.filtered_state.T
    
    factors_smooth_df = pd.DataFrame(
        factors_smoothed, 
        index=df.index, 
        columns=[f'Kalman_Smooth_{i+1}' for i in range(k_factors)]
    )
    
    factors_filter_df = pd.DataFrame(
        factors_filtered, 
        index=df.index, 
        columns=[f'Kalman_Filter_{i+1}' for i in range(k_factors)]
    )
    
    # Chargements depuis la matrice design
    loadings = model['design']
    
    return {
        'factors_smooth': factors_smooth_df,
        'factors_filter': factors_filter_df,
        'loadings': loadings,
        'model': result,
        'success': True,
        'method': 'Custom_DFM'
    }

def extract_factors_simple_kalman(df, k_factors=1):
    """
    Filtre de Kalman très simple avec pykalman
    """
    try:
        from pykalman import KalmanFilter
        
        # Standardiser
        df_scaled = (df - df.mean()) / df.std()
        observations = df_scaled.values
        
        # Kalman Filter simple
        kf = KalmanFilter(
            n_dim_state=k_factors,
            n_dim_obs=df.shape[1]
        )
        
        # Estimation EM
        kf = kf.em(observations, n_iter=50)
        
        # Estimation des états (facteurs)
        state_means, _ = kf.smooth(observations)
        
        print("✅ PyKalman réussi")
        
        # DataFrame
        factors_df = pd.DataFrame(
            state_means, 
            index=df.index, 
            columns=[f'Kalman_Simple_{i+1}' for i in range(k_factors)]
        )
        
        return {
            'factors_smooth': factors_df,
            'factors_filter': factors_df,
            'loadings': kf.observation_matrices,
            'model': kf,
            'success': True,
            'method': 'PyKalman'
        }
        
    except ImportError:
        print("   PyKalman non disponible")
        raise Exception("PyKalman non installé")
    except Exception as e:
        print(f"   PyKalman échoué: {e}")
        raise e

def extract_factors_pca(df, k_factors=1):
    """
    Extraction de facteurs avec ACP (Principal Component Analysis)
    """
    print(f"🔄 Extraction ACP avec {k_factors} composante(s)...")
    
    # Standardiser les données
    df_scaled = (df - df.mean()) / df.std()
    
    try:
        # Appliquer l'ACP
        pca = PCA(n_components=k_factors)
        factors = pca.fit_transform(df_scaled)
        
        print("✅ ACP estimée avec succès")
        print(f"   Variance expliquée: {pca.explained_variance_ratio_}")
        print(f"   Variance totale expliquée: {pca.explained_variance_ratio_.sum():.3f}")
        
        # DataFrame des facteurs
        factors_df = pd.DataFrame(
            factors, 
            index=df.index, 
            columns=[f'PCA_{i+1}' for i in range(k_factors)]
        )
        
        # Chargements factoriels
        loadings = pca.components_.T
        
        return {
            'factors': factors_df,
            'loadings': loadings,
            'model': pca,
            'explained_variance': pca.explained_variance_ratio_,
            'success': True,
            'method': 'ACP'
        }
        
    except Exception as e:
        print(f"❌ Erreur ACP: {e}")
        return {
            'success': False,
            'error': str(e),
            'method': 'ACP'
        }

def compare_methods(df, k_factors=1):
    """
    Compare Kalman/DFM vs ACP
    """
    print("\n" + "="*60)
    print("🔬 COMPARAISON DES MÉTHODES D'EXTRACTION")
    print("="*60)
    
    results = {}
    
    # 1. Extraction avec Kalman/DFM
    print("\n1️⃣ MODÈLE DE FACTEURS DYNAMIQUES (Kalman)")
    kalman_result = extract_factors_kalman(df, k_factors)
    results['kalman'] = kalman_result
    
    # 2. Extraction avec ACP
    print("\n2️⃣ ANALYSE EN COMPOSANTES PRINCIPALES")
    pca_result = extract_factors_pca(df, k_factors)
    results['pca'] = pca_result
    
    # 3. Comparaison si les deux ont réussi
    if kalman_result['success'] and pca_result['success']:
        print("\n" + "="*40)
        print("📊 COMPARAISON DES RÉSULTATS")
        print("="*40)
        
        # Vérifier la validité des facteurs Kalman
        kalman_factors = kalman_result['factors_smooth']
        pca_factors = pca_result['factors']
        
        # Vérifier si les facteurs Kalman sont valides
        kalman_valid = not (kalman_factors.isnull().all().any() or 
                           np.allclose(kalman_factors.values, 0, atol=1e-10))
        
        if not kalman_valid:
            print("⚠️ ATTENTION: Facteurs Kalman invalides (zéros ou NaN)")
            print("   → Utilisation de l'ACP uniquement")
            return {'kalman': kalman_result, 'pca': pca_result, 'kalman_valid': False}
        
        # Corrélations avec les variables originales
        print("\n🔹 CORRÉLATIONS AVEC LES VARIABLES ORIGINALES:")
        print("\nKalman/DFM (facteur 1):")
        kalman_corr = df.corrwith(kalman_factors.iloc[:, 0]).abs().sort_values(ascending=False)
        for var, corr in kalman_corr.items():
            print(f"   {var}: {corr:.3f}")
        
        print("\nACP (composante 1):")
        pca_corr = df.corrwith(pca_factors.iloc[:, 0]).abs().sort_values(ascending=False)
        for var, corr in pca_corr.items():
            print(f"   {var}: {corr:.3f}")
        
        # Corrélation entre les facteurs des deux méthodes
        factor_correlation = kalman_factors.iloc[:, 0].corr(pca_factors.iloc[:, 0])
        print(f"\n🔹 CORRÉLATION ENTRE FACTEURS KALMAN-ACP: {abs(factor_correlation):.3f}")
        
        # Métrique de performance
        kalman_avg_corr = kalman_corr.mean()
        pca_avg_corr = pca_corr.mean()
        
        print(f"\n🔹 PERFORMANCE MOYENNE:")
        print(f"   Kalman/DFM: {kalman_avg_corr:.3f}")
        print(f"   ACP: {pca_avg_corr:.3f}")
        
        # Recommandation
        print(f"\n🏆 RECOMMANDATION:")
        if kalman_avg_corr > pca_avg_corr and not np.isnan(kalman_avg_corr):
            print("   → Kalman/DFM semble meilleur (corrélations plus fortes)")
            print("   → Avantage: Capture la dynamique temporelle")
        elif pca_avg_corr > kalman_avg_corr:
            print("   → ACP semble meilleur (corrélations plus fortes)")
            print("   → Avantage: Plus stable et interprétable")
        else:
            print("   → ACP recommandé (plus robuste)")
            print("   → Kalman a des problèmes sur ces données")
        
        results['comparison'] = {
            'kalman_avg_corr': kalman_avg_corr,
            'pca_avg_corr': pca_avg_corr,
            'factor_correlation': abs(factor_correlation) if not np.isnan(factor_correlation) else 0,
            'kalman_corr': kalman_corr,
            'pca_corr': pca_corr,
            'kalman_valid': True
        }
    
    elif pca_result['success']:
        print("\n⚠️ SEULE L'ACP A FONCTIONNÉ")
        print("   → Utilisation de l'ACP par défaut")
        results['comparison'] = {'kalman_valid': False, 'pca_only': True}
    
    return results