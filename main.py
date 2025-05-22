import pandas as pd
import matplotlib.pyplot as plt
import os
from code.data_load import load_data, clean_data
from code.kalman_filter import compare_methods

def analyze_economic_factor(correlations):
    """
    Analyse économique du facteur principal
    """
    print("  Interprétation:")
    
    # Analyser chaque variable
    for var, corr in correlations.items():
        if abs(corr) > 0.5:  # Corrélation forte
            if var == 'Climat_Affaires':
                if corr > 0:
                    print(f"  → Facteur positivement lié au climat des affaires ({corr:.3f})")
                else:
                    print(f"  → Facteur négativement lié au climat des affaires ({corr:.3f})")
                    
            elif var == 'Taux_Chomage':
                if corr < 0:
                    print(f"  → Facteur anti-corrélé au chômage = bon signe économique ({corr:.3f})")
                else:
                    print(f"  → Facteur corrélé au chômage = stress économique ({corr:.3f})")
                    
            elif var == 'Indice_Prix_Conso':
                if corr > 0:
                    print(f"  → Facteur capte les pressions inflationnistes ({corr:.3f})")
                else:
                    print(f"  → Facteur anti-corrélé à l'inflation ({corr:.3f})")
                    
            elif var == 'Taux_10ans_FR':
                if corr > 0:
                    print(f"  → Facteur suit les taux longs (stress financier) ({corr:.3f})")
                else:
                    print(f"  → Facteur inversement lié aux taux longs ({corr:.3f})")
                    
            elif var == 'Indicateur_production_indus':
                if corr > 0:
                    print(f"  → Facteur lié positivement à la production ({corr:.3f})")
                else:
                    print(f"  → Facteur inversement lié à la production ({corr:.3f})")
    
    # Conclusion générale
    stress_indicators = ['Taux_Chomage', 'Taux_10ans_FR']
    growth_indicators = ['Climat_Affaires', 'Indicateur_production_indus']
    
    stress_corr = sum([correlations.get(var, 0) for var in stress_indicators if var in correlations])
    growth_corr = sum([correlations.get(var, 0) for var in growth_indicators if var in correlations])
    
    print(f"\n  💡 Conclusion:")
    if stress_corr > 0 and growth_corr < 0:
        print("     Le facteur semble capturer un 'STRESS ÉCONOMIQUE'")
    elif stress_corr < 0 and growth_corr > 0:
        print("     Le facteur semble capturer la 'CROISSANCE ÉCONOMIQUE'")
    else:
        print("     Le facteur capture un mélange complexe d'indicateurs")

def create_comparison_visualizations(results, save_path="output/visualisations/"):
    """
    Crée des graphiques comparatifs
    """
    import matplotlib.pyplot as plt
    import os
    os.makedirs(save_path, exist_ok=True)
    
    if results['kalman']['success'] and results['pca']['success']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Facteurs Kalman Smoothed
        kalman_factors = results['kalman']['factors_smooth']
        axes[0,0].plot(kalman_factors.index, kalman_factors.iloc[:, 0], 'b-', linewidth=2)
        axes[0,0].set_title('Facteur 1 - Kalman/DFM (Smoothed)', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Facteurs Kalman Filtered vs Smoothed
        kalman_filtered = results['kalman']['factors_filter']
        axes[0,1].plot(kalman_factors.index, kalman_factors.iloc[:, 0], 'b-', label='Smoothed', linewidth=2)
        axes[0,1].plot(kalman_filtered.index, kalman_filtered.iloc[:, 0], 'r--', label='Filtered', linewidth=2)
        axes[0,1].set_title('Kalman: Filtered vs Smoothed', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Facteurs ACP
        pca_factors = results['pca']['factors']
        axes[1,0].plot(pca_factors.index, pca_factors.iloc[:, 0], 'g-', linewidth=2)
        axes[1,0].set_title('Facteur 1 - ACP', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # Comparaison directe
        axes[1,1].plot(kalman_factors.index, kalman_factors.iloc[:, 0], 'b-', label='Kalman/DFM', linewidth=2)
        axes[1,1].plot(pca_factors.index, pca_factors.iloc[:, 0], 'g-', label='ACP', linewidth=2)
        axes[1,1].set_title('Comparaison Kalman vs ACP', fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}comparison_kalman_pca.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Graphiques comparatifs sauvegardés: {save_path}comparison_kalman_pca.png")
    
    elif results['kalman']['success']:
        kalman_factors = results['kalman']['factors_smooth']
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(kalman_factors.columns):
            plt.plot(kalman_factors.index, kalman_factors[col], label=col, linewidth=2)
        plt.title('Facteurs Kalman/DFM', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}factors_kalman.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    elif results['pca']['success']:
        pca_factors = results['pca']['factors']
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(pca_factors.columns):
            plt.plot(pca_factors.index, pca_factors[col], label=col, linewidth=2)
        plt.title('Facteurs ACP', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}factors_pca.png", dpi=300, bbox_inches='tight')
        plt.show()
    """
    Crée les graphiques des facteurs
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Graphique évolution des facteurs
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(factors_df.columns):
        plt.plot(factors_df.index, factors_df[col], label=col, linewidth=2)
    
    plt.title('Évolution des Facteurs Extraits', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Valeur du facteur')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{save_path}factors_evolution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Graphique sauvegardé: {save_path}factors_evolution.png")

def save_comparison_results(results, original_df):
    """
    Sauvegarde les résultats comparatifs
    """
    import os
    os.makedirs("output", exist_ok=True)
    
    # Sauvegarder les facteurs de chaque méthode
    if results['kalman']['success']:
        results['kalman']['factors_smooth'].to_csv("output/factors_kalman_smooth.csv")
        results['kalman']['factors_filter'].to_csv("output/factors_kalman_filter.csv")
        print("✓ Facteurs Kalman sauvegardés: output/factors_kalman_*.csv")
    
    if results['pca']['success']:
        results['pca']['factors'].to_csv("output/factors_pca.csv")
        print("✓ Facteurs ACP sauvegardés: output/factors_pca.csv")
    
    # Rapport comparatif
    with open("output/rapport_comparaison.txt", "w", encoding='utf-8') as f:
        f.write("=== RAPPORT DE COMPARAISON KALMAN vs ACP ===\n\n")
        f.write(f"Nombre de facteurs extraits: 2\n")
        f.write(f"Nombre d'observations: {original_df.shape[0]}\n")
        f.write(f"Variables analysées: {list(original_df.columns)}\n\n")
        
        if results['kalman']['success']:
            f.write("RÉSULTATS KALMAN/DFM:\n")
            f.write(f"✓ Succès: Oui\n")
            f.write(f"✓ Log-vraisemblance: {results['kalman']['model'].llf:.2f}\n")
            f.write(f"✓ AIC: {results['kalman']['model'].aic:.2f}\n")
            f.write(f"✓ BIC: {results['kalman']['model'].bic:.2f}\n\n")
        else:
            f.write("RÉSULTATS KALMAN/DFM:\n")
            f.write(f"❌ Échec: {results['kalman']['error']}\n\n")
        
        if results['pca']['success']:
            f.write("RÉSULTATS ACP:\n")
            f.write(f"✓ Succès: Oui\n")
            f.write(f"✓ Variance expliquée facteur 1: {results['pca']['explained_variance'][0]:.3f}\n")
            f.write(f"✓ Variance expliquée facteur 2: {results['pca']['explained_variance'][1]:.3f}\n")
            f.write(f"✓ Variance totale expliquée: {results['pca']['explained_variance'].sum():.3f}\n\n")
        
        if 'comparison' in results:
            comp = results['comparison']
            f.write("COMPARAISON:\n")
            f.write(f"✓ Corrélation moyenne Kalman: {comp['kalman_avg_corr']:.3f}\n")
            f.write(f"✓ Corrélation moyenne ACP: {comp['pca_avg_corr']:.3f}\n")
            f.write(f"✓ Corrélation entre facteurs: {comp['factor_correlation']:.3f}\n\n")
            
            if comp['kalman_avg_corr'] > comp['pca_avg_corr']:
                f.write("RECOMMANDATION: Kalman/DFM (corrélations plus fortes)\n")
            else:
                f.write("RECOMMANDATION: ACP (corrélations plus fortes ou plus stable)\n")
    
    print("✓ Rapport comparatif: output/rapport_comparaison.txt")
    """
    Sauvegarde les résultats
    """
    os.makedirs("output", exist_ok=True)
    
    # Sauvegarder les facteurs
    factors_df.to_csv("output/factors.csv")
    print("✓ Facteurs sauvegardés: output/factors.csv")
    
    # Calculer les corrélations
    correlations = original_df.corrwith(factors_df.iloc[:, 0]).sort_values(ascending=False)
    
    # Rapport simple
    with open("output/rapport_facteurs.txt", "w") as f:
        f.write("=== RAPPORT D'EXTRACTION DE FACTEURS ===\n\n")
        f.write(f"Nombre de facteurs extraits: {factors_df.shape[1]}\n")
        f.write(f"Nombre d'observations: {factors_df.shape[0]}\n")
        f.write(f"Période: {factors_df.index[0]} à {factors_df.index[-1]}\n\n")
        
        f.write("CORRÉLATIONS AVEC LE PREMIER FACTEUR:\n")
        for var, corr in correlations.items():
            f.write(f"{var}: {corr:.3f}\n")
        
        f.write(f"\nSTATISTIQUES DU PREMIER FACTEUR:\n")
        f.write(f"Moyenne: {factors_df.iloc[:, 0].mean():.3f}\n")
        f.write(f"Écart-type: {factors_df.iloc[:, 0].std():.3f}\n")
        f.write(f"Min: {factors_df.iloc[:, 0].min():.3f}\n")
        f.write(f"Max: {factors_df.iloc[:, 0].max():.3f}\n")
    
    print("✓ Rapport sauvegardé: output/rapport_facteurs.txt")

def main():
    """
    Script principal d'analyse
    """
    print("="*60)
    print("    EXTRACTION DE FACTEURS - FILTRE DE KALMAN")
    print("="*60)
    
    # 1. Charger les données
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(f"\n❌ ARRÊT DU PROGRAMME: {e}")
        print("\nAssurez-vous que votre fichier 'donnees_mergees_complet.csv' est dans le dossier 'data/'")
        return None, None
    
    # 2. Nettoyer les données
    df_clean = clean_data(df)
    
    # 3. Afficher info sur les données
    print(f"\nDonnées finales:")
    
    # Vérifier le type d'index
    if hasattr(df_clean.index[0], 'strftime'):
        print(f"- Période: {df_clean.index[0].strftime('%Y-%m')} à {df_clean.index[-1].strftime('%Y-%m')}")
    else:
        print(f"- Observations: {df_clean.shape[0]} lignes")
    
    print(f"- Variables: {list(df_clean.columns)}")
    print(f"- Dimensions: {df_clean.shape}")
    
    # 4. Comparer les méthodes d'extraction
    k_factors = 2  # Nombre de facteurs à extraire
    results = compare_methods(df_clean, k_factors=k_factors)
    
    # 5. Analyser les résultats
    print("\n" + "="*40)
    print("📈 ANALYSE DES FACTEURS EXTRAITS")
    print("="*40)
    
    # Utiliser les meilleurs résultats disponibles
    if results['kalman']['success'] and results.get('comparison', {}).get('kalman_valid', True):
        factors = results['kalman']['factors_smooth']
        method_used = "Kalman/DFM (Smoothed)"
        print(f"✅ Utilisation des facteurs {method_used}")
    elif results['pca']['success']:
        factors = results['pca']['factors']
        method_used = "ACP"
        print(f"✅ Utilisation des facteurs {method_used}")
        if results['kalman']['success']:
            print("   ℹ️ Facteurs Kalman invalides → fallback vers ACP")
    else:
        print("❌ Aucune méthode n'a fonctionné")
        return None, None
    
    print(f"\nPremières valeurs des facteurs ({method_used}):")
    print(factors.head())
    
    # Corrélations avec variables originales
    print(f"\nCorrélations du premier facteur avec les variables économiques:")
    correlations = df_clean.corrwith(factors.iloc[:, 0]).sort_values(key=abs, ascending=False)
    for var, corr in correlations.items():
        print(f"  {var}: {corr:.3f}")
    
    # Analyse économique spécifique
    print(f"\nInterprétation économique du facteur principal:")
    analyze_economic_factor(correlations)
    
    # 6. Créer visualisations comparatives
    print("\n" + "="*40)
    print("📊 VISUALISATIONS COMPARATIVES")
    print("="*40)
    
    create_comparison_visualizations(results)
    
    # 7. Sauvegarder résultats
    print("\n" + "="*40)
    print("SAUVEGARDE")
    print("="*40)
    
    save_comparison_results(results, df_clean)
    
    # 8. Résumé final
    print("\n" + "="*60)
    print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
    print("="*60)
    
    print("\nFichiers générés:")
    print("- output/factors.csv")
    print("- output/rapport_facteurs.txt") 
    print("- output/visualisations/factors_evolution.png")
    
    print(f"\nLe facteur principal explique les variations communes de:")
    # Les vraies variables de votre fichier
    variables_importantes = ['Climat_Affaires', 'Taux_Chomage', 'Indice_Prix_Conso', 
                           'Indicateur_production_indus', 'Taux_10ans_FR']
    top_corr = correlations.head(3)
    for var, corr in top_corr.items():
        print(f"  • {var} (corrélation: {corr:.3f})")
    
    return factors, df_clean

if __name__ == "__main__":
    results, data = main()