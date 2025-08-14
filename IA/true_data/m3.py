# m3_segmentation_fixed.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

INCIDENTS_CSV = "incidents.csv"
LOGINS_CSV = "logins.csv"

def load_and_clean_data():
    """Load and clean data, handling empty rows."""
    try:
        # Load data with empty row handling
        inc = pd.read_csv(INCIDENTS_CSV, parse_dates=["Date"], skip_blank_lines=True).dropna(how='all')
        log = pd.read_csv(LOGINS_CSV, parse_dates=["DateHeure"], skip_blank_lines=True).dropna(how='all')
        
        # Remove rows with critical missing values
        if not inc.empty:
            inc = inc.dropna(subset=["Date", "Entreprise"])
            print(f"Loaded {len(inc)} incidents after cleaning")
        
        if not log.empty:
            log = log.dropna(subset=["DateHeure", "Utilisateur"])
            # Standardize login results
            log["Resultat"] = log["Resultat"].fillna("unknown").str.lower().str.strip()
            print(f"Loaded {len(log)} login records after cleaning")
        
        return inc, log
        
    except FileNotFoundError as e:
        print(f"Erreur: Fichier non trouvé - {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return pd.DataFrame(), pd.DataFrame()

inc, log = load_and_clean_data()

# ---------- Segmentation des entreprises ----------
if not inc.empty and len(inc) > 0:
    try:
        # Fill missing values before aggregation
        inc["ImpactAriary"] = inc["ImpactAriary"].fillna(0)
        inc["IndispoHeures"] = inc["IndispoHeures"].fillna(0)
        inc["TypeAttaque"] = inc["TypeAttaque"].fillna("unknown")
        inc["Secteur"] = inc["Secteur"].fillna("Unknown")
        inc["Taille"] = inc["Taille"].fillna(inc["Taille"].median())
        
        # Features: fréquence d'incidents, impact moyen, indispo moyenne, diversité des attaques
        agg_ent = (
            inc.groupby("Entreprise")
              .agg(
                  freq_incidents=("Entreprise","size"),
                  impact_moy=("ImpactAriary","mean"),
                  indispo_moy=("IndispoHeures","mean"),
                  nb_types=("TypeAttaque","nunique"),
                  secteur=("Secteur","first"),
                  taille=("Taille","first")
              )
              .reset_index()
        )
        
        # Handle any remaining NaN values
        agg_ent = agg_ent.fillna(0)
        
        # Only proceed if we have enough data for clustering
        if len(agg_ent) >= 3:
            X_ent = agg_ent[["freq_incidents","impact_moy","indispo_moy","nb_types","taille"]]
            
            # Check for non-finite values
            X_ent = X_ent.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            scaler_ent = StandardScaler()
            X_ent_scaled = scaler_ent.fit_transform(X_ent)
            
            # Adjust number of clusters based on data size
            n_clusters = min(3, len(agg_ent))
            k_ent = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            agg_ent["cluster_esn"] = k_ent.fit_predict(X_ent_scaled)
            
            # PCA 2D pour visualiser
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_ent_scaled)
            agg_ent["pc1"] = coords[:,0]
            agg_ent["pc2"] = coords[:,1]
            
            print("=== Segments entreprises ===")
            print(agg_ent[["Entreprise","secteur","taille","freq_incidents","impact_moy","indispo_moy","nb_types","cluster_esn"]].head())
            
            agg_ent.to_csv("segments_entreprises.csv", index=False)
            print("Segments entreprises sauvegardés dans segments_entreprises.csv")
            
        else:
            print("Pas assez de données d'entreprises pour effectuer la segmentation")
            # Create empty file to avoid errors in other modules
            pd.DataFrame().to_csv("segments_entreprises.csv", index=False)
            
    except Exception as e:
        print(f"Erreur lors de la segmentation des entreprises: {e}")
        pd.DataFrame().to_csv("segments_entreprises.csv", index=False)
else:
    print("Aucune donnée d'incident disponible pour la segmentation des entreprises")
    pd.DataFrame().to_csv("segments_entreprises.csv", index=False)

# ---------- Segmentation des utilisateurs ----------
if not log.empty and len(log) > 0:
    try:
        # Corriger les labels de résultat selon les données réelles
        # Supposons que les valeurs peuvent être: "succès", "échec", "success", "failure", etc.
        log["is_fail"] = log["Resultat"].isin(["échec", "failure", "fail"]).astype(int)
        log["is_succ"] = log["Resultat"].isin(["succès", "success"]).astype(int)
        
        # Fill missing values
        log["Localisation"] = log["Localisation"].fillna("Unknown")
        log["IPSource"] = log["IPSource"].fillna("0.0.0.0")
        log["Role"] = log["Role"].fillna("Unknown")
        log["Departement"] = log["Departement"].fillna("Unknown")
        
        # Construire des features par utilisateur
        agg_user = (
            log.groupby(["Utilisateur","Role","Departement"])
               .agg(
                   nb_echecs=("is_fail","sum"),
                   nb_succes=("is_succ","sum"),
                   nb_total=("Resultat","size"),
                   nb_pays=("Localisation","nunique"),
                   nb_ip=("IPSource","nunique")
               ).reset_index()
        )
        
        # Handle division by zero
        agg_user["ratio_echec"] = np.where(
            agg_user["nb_total"] > 0, 
            agg_user["nb_echecs"] / agg_user["nb_total"], 
            0
        )
        
        # Handle any remaining NaN or inf values
        agg_user = agg_user.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Only proceed if we have enough data for clustering
        if len(agg_user) >= 4:
            X_user = agg_user[["nb_echecs","nb_succes","nb_total","nb_pays","nb_ip","ratio_echec"]]
            
            # Ensure no non-finite values
            X_user = X_user.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            scaler_usr = StandardScaler()
            X_user_scaled = scaler_usr.fit_transform(X_user)
            
            # Adjust number of clusters based on data size
            n_clusters = min(4, len(agg_user))
            k_usr = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            agg_user["cluster_risque"] = k_usr.fit_predict(X_user_scaled)
            
            print("\n=== Segments utilisateurs ===")
            print(agg_user.head())
            
            agg_user.to_csv("segments_utilisateurs.csv", index=False)
            print("Segments utilisateurs sauvegardés dans segments_utilisateurs.csv")
            
        else:
            print("Pas assez de données d'utilisateurs pour effectuer la segmentation")
            # Create empty file to avoid errors in other modules
            pd.DataFrame().to_csv("segments_utilisateurs.csv", index=False)
            
    except Exception as e:
        print(f"Erreur lors de la segmentation des utilisateurs: {e}")
        pd.DataFrame().to_csv("segments_utilisateurs.csv", index=False)
else:
    print("Aucune donnée de login disponible pour la segmentation des utilisateurs")
    pd.DataFrame().to_csv("segments_utilisateurs.csv", index=False)