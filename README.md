# Dynamic Factor Models & Kalman Filtering 

**Objectif :** Reproduire la méthodologie de l’article *Factor Extraction using Kalman Filter and Smoothing* (Poncela et al., 2021) sur un jeu de données économiques françaises, et comparer les résultats avec l’Analyse en Composantes Principales (ACP).

---

## 📁 Structure du projet

```text
├── data/                          # Données macroéconomiques traitées
│   └── donnees_mergees_complet.csv
├── code/
│   ├── kalman_filter.py           # Implémentation du DFM et extraction des facteurs
│   ├── data_load.py               # Chargement et nettoyage des données
├── output/                        # Résultats générés (facteurs, graphiques, rapports)
│   ├── factors_kalman_smooth.csv
│   ├── factors_pca.csv
│   ├── visualisations/
│   └── rapport_comparaison.txt
├── main.py                        # Script principal d'exécution
├── README.md                      # Présentation du projet
├── requirements.txt               # Dépendances Python
```


---

## 📦 Installation

## Travailler sur le Projet

1️⃣ Cloner le projet
```bash
git clone https://github.com/morganjowitt/eco_fi.git
```
Ensuite se positionner où il y a le fichier cloné

```bash
cd eco_fi
```
2️⃣ Création d'un environment virtuel

```bash
python3 -m venv .venv
```
3️⃣ Activer l'environnement virtuel

```bash
source .venv/bin/activate   # Sur Windows : .venv\Scripts\activate
```
4️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

## 👥 Authors
- [Aya MOKHTAR](https://github.com/ayamokhtar)
- [Morgan Jowitt](https://github.com/morganjowitt)
- [Gaétan Dumas](https://github.com/gaetan250)
- [Pierre Liberge](https://github.com/pierreliberge)
