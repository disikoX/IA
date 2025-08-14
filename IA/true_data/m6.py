# m6_prediction_corrige_fixed.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def load_and_clean_data():
    """Load and clean data, handling empty rows."""
    try:
        # Load data with empty row handling
        inc = pd.read_csv("incidents.csv", parse_dates=["Date"], skip_blank_lines=True).dropna(how='all')
        log = pd.read_csv("logins.csv", parse_dates=["DateHeure"], skip_blank_lines=True).dropna(how='all')
        
        # Clean incidents data
        if not inc.empty:
            inc = inc.dropna(subset=["Date", "Entreprise"])
            inc["ImpactAriary"] = inc["ImpactAriary"].fillna(0)
            inc["IndispoHeures"] = inc["IndispoHeures"].fillna(0)
            print(f"Loaded {len(inc)} incidents after cleaning")
        
        # Clean logins data
        if not log.empty:
            log = log.dropna(subset=["DateHeure", "Utilisateur"])
            log["Resultat"] = log["Resultat"].fillna("unknown").str.lower().str.strip()
            log["IPSource"] = log["IPSource"].fillna("0.0.0.0")
            log["Localisation"] = log["Localisation"].fillna("Unknown")
            log["Role"] = log["Role"].fillna("Unknown")
            log["Departement"] = log["Departement"].fillna("Unknown")
            print(f"Loaded {len(log)} login records after cleaning")
        
        return inc, log
        
    except FileNotFoundError as e:
        print(f"Erreur: Fichier non trouvé - {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return pd.DataFrame(), pd.DataFrame()

inc, log = load_and_clean_data()

# ---------- PARTIE 1 : Entreprises ----------
if not inc.empty and len(inc) >= 10:  # Need minimum data for prediction
    try:
        # Construire une série mensuelle par entreprise
        g = inc.groupby("Entreprise")
        frames = []

        for ent, df in g:
            if len(df) >= 2:  # Need at least 2 records per enterprise
                s = df.set_index("Date").resample("ME").agg({
                    "Entreprise": "size",
                    "ImpactAriary": "sum",
                    "IndispoHeures": "sum"
                }).rename(columns={
                    "Entreprise": "nb_incidents",
                    "ImpactAriary": "impact_total",
                    "IndispoHeures": "indispo_total"
                }).fillna(0)
                s["Entreprise"] = ent
                frames.append(s.reset_index())

        if frames:
            panel = pd.concat(frames, ignore_index=True)

            # Lag features + cible
            panel = panel.sort_values(["Entreprise","Date"])
            panel["nb_incidents_prev1"] = panel.groupby("Entreprise")["nb_incidents"].shift(1).fillna(0)
            panel["impact_prev1"] = panel.groupby("Entreprise")["impact_total"].shift(1).fillna(0)
            panel["indispo_prev1"] = panel.groupby("Entreprise")["indispo_total"].shift(1).fillna(0)

            # Cible: y=1 si nb_incidents du mois courant >0
            panel["y_current"] = (panel["nb_incidents"] > 0).astype(int)
            panel["y_next"] = panel.groupby("Entreprise")["y_current"].shift(-1)
            panel = panel.dropna(subset=["y_next"])

            if len(panel) >= 10:  # Need minimum samples for ML
                X_ent = panel[["nb_incidents_prev1","impact_prev1","indispo_prev1"]]
                y_ent = panel["y_next"].astype(int)

                # Check if we have both classes
                if len(y_ent.unique()) > 1 and len(X_ent) >= 10:
                    # Split train/test
                    test_size = min(0.25, max(0.1, len(X_ent) // 10))  # Adaptive test size
                    try:
                        Xtr, Xte, ytr, yte = train_test_split(
                            X_ent, y_ent, 
                            test_size=test_size, 
                            random_state=42, 
                            stratify=y_ent
                        )
                    except ValueError:
                        # If stratify fails, use random split
                        Xtr, Xte, ytr, yte = train_test_split(
                            X_ent, y_ent, 
                            test_size=test_size, 
                            random_state=42
                        )

                    # RandomForest pour prédire incidents prochain mois
                    rf = RandomForestClassifier(
                        n_estimators=min(100, len(Xtr) * 2), 
                        random_state=42, 
                        class_weight="balanced"
                    )
                    rf.fit(Xtr, ytr)
                    pred_proba = rf.predict_proba(Xte)[:,1]
                    pred = (pred_proba >= 0.5).astype(int)
                    
                    print("=== Entreprises: RandomForest ===")
                    print(classification_report(yte, pred, digits=3, zero_division=0))
                    print("AUC:", roc_auc_score(yte, pred_proba))
                else:
                    print("Données insuffisantes ou une seule classe pour la prédiction entreprises")
            else:
                print("Pas assez d'échantillons pour l'entraînement du modèle entreprises")
        else:
            print("Aucune donnée valide pour construire le panel entreprises")
    except Exception as e:
        print(f"Erreur lors de la prédiction entreprises: {e}")
else:
    print("Données insuffisantes pour la prédiction des incidents d'entreprises")

# ---------- PARTIE 2 : Utilisateurs ----------
if not log.empty and len(log) >= 10:  # Need minimum data for prediction
    try:
        # Correct result labeling based on actual data
        log["is_fail"] = log["Resultat"].isin(["échec", "failure", "fail"]).astype(int)
        log["is_succ"] = log["Resultat"].isin(["succès", "success"]).astype(int)

        # Label proxy "compromis": échec suivi d'un succès depuis IP différente <1h
        log = log.sort_values(["Utilisateur","DateHeure"])
        log["IP_prev"] = log.groupby("Utilisateur")["IPSource"].shift(1)
        log["Date_prev"] = log.groupby("Utilisateur")["DateHeure"].shift(1)
        log["Result_prev"] = log.groupby("Utilisateur")["is_fail"].shift(1)

        def compromised_row(row):
            if pd.isna(row["Date_prev"]): 
                return 0
            dt = (row["DateHeure"] - row["Date_prev"]).total_seconds() / 3600.0
            return int((row["Result_prev"] == 1) and (row["is_succ"] == 1) 
                       and (row["IP_prev"] != row["IPSource"]) and (dt <= 1.0))

        log["compromis_signal"] = log.apply(compromised_row, axis=1)

        # Agrégation par utilisateur
        agg_user = (
            log.groupby(["Utilisateur","Role","Departement"])
               .agg(
                   nb_echecs=("is_fail","sum"),
                   nb_total=("Resultat","size"),
                   nb_ip=("IPSource","nunique"),
                   nb_pays=("Localisation","nunique"),
                   nb_compromis=("compromis_signal","sum")
               ).reset_index()
        )
        agg_user["ratio_echec"] = np.where(
            agg_user["nb_total"] > 0, 
            agg_user["nb_echecs"] / agg_user["nb_total"], 
            0
        )
        agg_user["y_compromis"] = (agg_user["nb_compromis"] > 0).astype(int)

        # Clean any remaining NaN or inf values
        agg_user = agg_user.replace([np.inf, -np.inf], np.nan).fillna(0)

        if len(agg_user) >= 10:  # Need minimum samples
            Xu = agg_user[["nb_echecs","nb_total","nb_ip","nb_pays","ratio_echec"]]
            yu = agg_user["y_compromis"]

            # Check if we have both classes and enough data
            if len(yu.unique()) > 1 and len(Xu) >= 10:
                test_size = min(0.25, max(0.1, len(Xu) // 10))  # Adaptive test size
                try:
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xu, yu, 
                        test_size=test_size, 
                        random_state=42, 
                        stratify=yu
                    )
                except ValueError:
                    # If stratify fails, use random split
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xu, yu, 
                        test_size=test_size, 
                        random_state=42
                    )

                lr = LogisticRegression(max_iter=200, class_weight="balanced")
                lr.fit(Xtr, ytr)
                proba = lr.predict_proba(Xte)[:,1]
                pred = (proba >= 0.5).astype(int)

                print("\n=== Utilisateurs: LogisticRegression ===")
                print(classification_report(yte, pred, digits=3, zero_division=0))
                print("AUC:", roc_auc_score(yte, proba))

                # Sauvegarde scores de risque
                agg_user["risk_score"] = lr.predict_proba(Xu)[:,1]
                risk_output = agg_user[["Utilisateur","Role","Departement","risk_score"]]
                risk_output.to_csv("risque_utilisateur.csv", index=False)
                print("\nScores de risque écrits dans risque_utilisateur.csv")
            else:
                print("Données insuffisantes ou une seule classe pour la prédiction utilisateurs")
                # Create empty risk file to avoid errors in other modules
                pd.DataFrame(columns=["Utilisateur","Role","Departement","risk_score"]).to_csv("risque_utilisateur.csv", index=False)
        else:
            print("Pas assez d'utilisateurs pour l'entraînement du modèle")
            pd.DataFrame(columns=["Utilisateur","Role","Departement","risk_score"]).to_csv("risque_utilisateur.csv", index=False)

    except Exception as e:
        print(f"Erreur lors de la prédiction utilisateurs: {e}")
        pd.DataFrame(columns=["Utilisateur","Role","Departement","risk_score"]).to_csv("risque_utilisateur.csv", index=False)
else:
    print("Données insuffisantes pour la prédiction de risque utilisateurs")
    pd.DataFrame(columns=["Utilisateur","Role","Departement","risk_score"]).to_csv("risque_utilisateur.csv", index=False)