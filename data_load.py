import pandas as pd

def charger_csv(path, nom_indicateur):
    df = pd.read_csv(path, skiprows=3, sep=";", names=["Période", nom_indicateur, "Code"])
    df = df.drop(columns=["Code"])
    return df

def expand_trimestres(df):
    trimestre_to_mois = {
        "T1": ["01", "02", "03"],
        "T2": ["04", "05", "06"],
        "T3": ["07", "08", "09"],
        "T4": ["10", "11", "12"]
    }
    lignes_expandees = []
    for _, row in df.iterrows():
        periode = row["Période"]
        valeur = row["Taux_Chomage"]
        if "T" in periode:
            annee, trim = periode.split("-")
            for mois in trimestre_to_mois[trim]:
                lignes_expandees.append({
                    "Période": f"{annee}-{mois}",
                    "Taux_Chomage": valeur
                })
        else:
            lignes_expandees.append({
                "Période": periode,
                "Taux_Chomage": valeur
            })
    return pd.DataFrame(lignes_expandees)

def main():
    # Chargement des indicateurs économiques
    df1 = charger_csv("input/serie_001565530_19052025/valeurs_mensuelles.csv", "Climat_Affaires")
    df2 = charger_csv("input/serie_001688527_19052025/valeurs_trimestrielles.csv", "Taux_Chomage")
    df3 = charger_csv("input/serie_001769682_19052025/valeurs_mensuelles.csv", "Indice_Prix_Conso")
    df4 = charger_csv("input/serie_010768261_19052025/valeurs_mensuelles.csv", "Indicateur_production_indus")

    # Expand des trimestres en mois
    df2_expanded = expand_trimestres(df2)

    # Fusion des DataFrames
    df_merged = df1.merge(df2_expanded, on="Période", how="outer") \
                   .merge(df3, on="Période", how="outer") \
                   .merge(df4, on="Période", how="outer")

    # Chargement des taux d'obligation
    df_oblig = pd.read_csv("input/WEBSTAT-observations-2025-05-19T23_11_39.031+02_00.csv",
                       skiprows=6,
                       sep=";",
                       names=["Date", "Taux_10ans_FR"])

    df_oblig["Date"] = pd.to_datetime(df_oblig["Date"], errors="coerce")
    df_oblig["Taux_10ans_FR"] = df_oblig["Taux_10ans_FR"].str.replace(",", ".", regex=False)
    df_oblig["Taux_10ans_FR"] = pd.to_numeric(df_oblig["Taux_10ans_FR"], errors="coerce")

    # Supprime les lignes vides ou mal lues
    df_oblig = df_oblig.dropna(subset=["Date", "Taux_10ans_FR"])
    df_oblig["Période"] = df_oblig["Date"].dt.strftime("%Y-%m")
    df_oblig = df_oblig[["Période", "Taux_10ans_FR"]]

    # Fusion finale
    df_merged_final = df_merged.merge(df_oblig, on="Période", how="outer")
    df_merged_final = df_merged_final.sort_values("Période").reset_index(drop=True)

    # Nettoyage des erreurs de Période
    df_merged_final = df_merged_final[~df_merged_final["Période"].isin(["Période"])]
    df_merged_final = df_merged_final.dropna(subset=["Période"])
    df_merged_final = df_merged_final.drop_duplicates()


    # Export CSV
    df_merged_final.to_csv("donnees_mergees_complet.csv", index=False, encoding="utf-8-sig")

    # Affichage d'un aperçu
    print(df_merged_final.tail(10))
    print(df_merged_final.shape)


if __name__ == "__main__":
    main()
