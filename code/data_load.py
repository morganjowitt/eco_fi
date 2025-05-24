import pandas as pd
import numpy as np

def load_data(file_path="data/donnees_mergees_complet.csv"):
    """
    Charge les données depuis le fichier CSV
    """
    print("Chargement des données...")
    
    try:
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if df.shape[1] > 1:
                    break
            except:
                continue
        
        print(f"✓ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        
        if 'Période' in df.columns:
            print("✓ Colonne 'Période' détectée")
            
            df = df.dropna(subset=['Période'])
            df = df[df['Période'] != 'Période']  
            
            try:
                df['Période'] = pd.to_datetime(df['Période'], format='%Y-%m', errors='coerce')
                df = df.dropna(subset=['Période'])
                df.set_index('Période', inplace=True)
                print(f"✓ Index temporel: {df.index[0].strftime('%Y-%m')} à {df.index[-1].strftime('%Y-%m')}")
            except Exception as e:
                print(f"Erreur conversion dates: {e}")
                print("✓ Conservation de l'index numérique")
        else:
            date_cols = [col for col in df.columns if any(word in col.lower() 
                        for word in ['date', 'time', 'periode'])]
            
            if date_cols:
                try:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    df = df.dropna(subset=[date_cols[0]])
                    df.set_index(date_cols[0], inplace=True)
                    print(f"✓ Index temporel: {df.index[0]} à {df.index[-1]}")
                except:
                    print("✓ Conservation de l'index numérique")
            else:
                print("ℹ Aucune colonne de dates détectée")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        
        print(f"✓ Variables numériques: {len(numeric_cols)}")
        
        return df
        
    except Exception as e:
        print(f"ERREUR: Impossible de charger le fichier {file_path}")
        print(f"Détails de l'erreur: {e}")
        print("\nVérifiez que:")
        print("1. Le fichier existe dans le dossier 'data/'")
        print("2. Le nom du fichier est correct: 'donnees_mergees_complet.csv'")
        print("3. Le fichier est au format CSV valide")
        
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable ou illisible")

def clean_data(df):
    """
    Nettoie les données (valeurs manquantes, outliers)
    """
    print("Nettoyage des données...")
    
    if df.isnull().any().any():
        print("⚠ Valeurs manquantes détectées - interpolation appliquée")
        df = df.interpolate().fillna(method='bfill').fillna(method='ffill')
    
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            print(f"⚠ {outliers} outliers détectés dans {col}")
            df[col] = df[col].clip(lower, upper)
    
    print(f"✓ Données nettoyées: {df.shape}")
    return df