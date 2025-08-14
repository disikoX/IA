# m7_dashboard_fixed.py
# Lance avec: streamlit run m7_dashboard_fixed.py
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from pathlib import Path

st.set_page_config(page_title="ESN Cybersécurité — Dashboard", layout="wide")

@st.cache_data
def load_data():
    """Load data with error handling for empty files."""
    def safe_load_csv(filename, parse_dates=None):
        try:
            if Path(filename).exists():
                df = pd.read_csv(filename, parse_dates=parse_dates, skip_blank_lines=True)
                df = df.dropna(how='all')  # Remove completely empty rows
                return df if not df.empty else None
            else:
                return None
        except Exception as e:
            st.warning(f"Erreur lors du chargement de {filename}: {e}")
            return None
    
    # Load each file with error handling
    inc = safe_load_csv("incidents.csv", parse_dates=["Date"])
    log = safe_load_csv("logins.csv", parse_dates=["DateHeure"])
    seg_ent = safe_load_csv("segments_entreprises.csv")
    seg_usr = safe_load_csv("segments_utilisateurs.csv")
    risk = safe_load_csv("risque_utilisateur.csv")
    
    # Clean the loaded data
    if inc is not None:
        inc = inc.dropna(subset=["Date"])
        inc["Secteur"] = inc["Secteur"].fillna("Unknown")
        inc["TypeAttaque"] = inc["TypeAttaque"].fillna("unknown")
        inc["ImpactAriary"] = inc["ImpactAriary"].fillna(0)
    
    if log is not None:
        log = log.dropna(subset=["DateHeure"])
        log["Resultat"] = log["Resultat"].fillna("unknown").str.lower().str.strip()
        log["Role"] = log["Role"].fillna("Unknown")
    
    return inc, log, seg_ent, seg_usr, risk

# Load data
inc, log, seg_ent, seg_usr, risk = load_data()

st.title("IAro Tech — Dashboard")

# Check if we have any data to work with
has_incidents = inc is not None and not inc.empty
has_logins = log is not None and not log.empty

if not has_incidents and not has_logins:
    st.error("Aucune donnée disponible. Vérifiez que les fichiers CSV existent et contiennent des données.")
    st.stop()

# Filtres (only show if data exists)
if has_incidents:
    secteurs = ["(Tous)"] + sorted(inc["Secteur"].dropna().unique().tolist())
    secteur_sel = st.sidebar.selectbox("Secteur", secteurs)
    type_att_options = sorted(inc["TypeAttaque"].dropna().unique().tolist())
    type_att_sel = st.sidebar.multiselect("Types d'attaque", type_att_options)
else:
    secteur_sel = "(Tous)"
    type_att_sel = []

if has_logins:
    role_options = sorted(log["Role"].dropna().unique().tolist())
    role_sel = st.sidebar.multiselect("Rôles", role_options)
else:
    role_sel = []

# Filter data based on selections
df_inc = inc.copy() if has_incidents else pd.DataFrame()
if has_incidents:
    if secteur_sel != "(Tous)":
        df_inc = df_inc[df_inc["Secteur"] == secteur_sel]
    if type_att_sel:
        df_inc = df_inc[df_inc["TypeAttaque"].isin(type_att_sel)]

df_log = log.copy() if has_logins else pd.DataFrame()
if has_logins and role_sel:
    df_log = df_log[df_log["Role"].isin(role_sel)]

# KPIs
col1, col2, col3 = st.columns(3)

with col1:
    incidents_count = len(df_inc) if has_incidents else 0
    st.metric("Incidents (filtrés)", incidents_count)

with col2:
    if has_incidents and not df_inc.empty and "ImpactAriary" in df_inc.columns:
        total_impact = df_inc["ImpactAriary"].sum()
        st.metric("Impact total (MGA)", f"{total_impact:,.0f}")
    else:
        st.metric("Impact total (MGA)", "N/A")

with col3:
    if has_logins and not df_log.empty:
        valid_results = df_log[df_log["Resultat"].isin(["échec", "succès", "success", "failure"])]
        if not valid_results.empty:
            failure_rate = valid_results["Resultat"].isin(["échec", "failure"]).mean()
            st.metric("Taux d'échec (logins)", f"{failure_rate:.2%}")
        else:
            st.metric("Taux d'échec (logins)", "N/A")
    else:
        st.metric("Taux d'échec (logins)", "N/A")

# Graph 1 : Incidents / mois
st.subheader("Incidents par mois")
if has_incidents and not df_inc.empty:
    try:
        inc_m = df_inc.set_index("Date").resample("M").size()
        if not inc_m.empty:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            inc_m.plot(ax=ax1, marker='o')
            ax1.set_title("Incidents par mois")
            ax1.set_xlabel("Mois")
            ax1.set_ylabel("Incidents")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
        else:
            st.info("Aucune donnée disponible pour le graphique des incidents mensuels")
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique des incidents: {e}")
else:
    st.info("Aucune donnée d'incident disponible")

# Graph 2 : Impact € par secteur (médiane)
st.subheader("Impact médian par secteur (MGA)")
if has_incidents and not df_inc.empty and "ImpactAriary" in df_inc.columns:
    try:
        # Filter valid data
        valid_impact = df_inc[
            (df_inc["Secteur"] != "Unknown") & 
            (df_inc["ImpactAriary"] > 0) & 
            df_inc["ImpactAriary"].notna()
        ]
        if not valid_impact.empty:
            imp_sec = valid_impact.groupby("Secteur")["ImpactAriary"].median().sort_values(ascending=False)
            if not imp_sec.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                imp_sec.plot(kind="bar", ax=ax2)
                ax2.set_title("Impact médian par secteur")
                ax2.set_xlabel("Secteur")
                ax2.set_ylabel("MGA médian")
                plt.xticks(rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("Aucune donnée valide pour l'impact par secteur")
        else:
            st.info("Aucune donnée d'impact valide disponible")
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique d'impact: {e}")
else:
    st.info("Aucune donnée d'impact disponible")

# Graph 3 : Taux d'échec des connexions (mensuel)
st.subheader("Taux d'échec des connexions (mensuel)")
if has_logins and not df_log.empty:
    try:
        # Filter valid login results
        valid_logins = df_log[df_log["Resultat"].isin(["échec", "succès", "success", "failure"])]
        if not valid_logins.empty:
            log_m = (valid_logins.set_index("DateHeure")
                    .resample("M")["Resultat"]
                    .apply(lambda s: s.isin(["échec", "failure"]).mean()))
            if not log_m.empty:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                log_m.plot(ax=ax3, marker='s')
                ax3.set_title("Taux d'échec (mensuel)")
                ax3.set_xlabel("Mois")
                ax3.set_ylabel("Taux")
                ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
            else:
                st.info("Aucune donnée disponible pour le taux d'échec mensuel")
        else:
            st.info("Aucune donnée de connexion valide disponible")
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique de taux d'échec: {e}")
else:
    st.info("Aucune donnée de connexion disponible")

# Tables segments / risques
st.subheader("Segments entreprises")
if seg_ent is not None and not seg_ent.empty:
    if "cluster_esn" in seg_ent.columns:
        try:
            st.dataframe(seg_ent.sort_values("cluster_esn"))
        except:
            st.dataframe(seg_ent)
    else:
        st.dataframe(seg_ent)
else:
    st.info("Aucune donnée de segmentation d'entreprises disponible")

st.subheader("Segments utilisateurs & risque")
if seg_usr is not None and not seg_usr.empty:
    if "cluster_risque" in seg_usr.columns:
        try:
            st.dataframe(seg_usr.sort_values("cluster_risque"))
        except:
            st.dataframe(seg_usr)
    else:
        st.dataframe(seg_usr)
else:
    st.info("Aucune donnée de segmentation d'utilisateurs disponible")

st.subheader("Top 50 utilisateurs à risque")
if risk is not None and not risk.empty:
    if "risk_score" in risk.columns:
        try:
            top_risk = risk.sort_values("risk_score", ascending=False).head(50)
            st.dataframe(top_risk)
        except:
            st.dataframe(risk.head(50))
    else:
        st.dataframe(risk.head(50))
else:
    st.info("Aucune donnée de score de risque disponible")

# Additional information panel
st.sidebar.markdown("---")
st.sidebar.subheader("Informations sur les données")
if has_incidents:
    st.sidebar.write(f"📊 Incidents chargés: {len(inc):,}")
if has_logins:
    st.sidebar.write(f"🔐 Connexions chargées: {len(log):,}")

if seg_ent is not None and not seg_ent.empty:
    st.sidebar.write(f"🏢 Entreprises segmentées: {len(seg_ent):,}")

if seg_usr is not None and not seg_usr.empty:
    st.sidebar.write(f"👤 Utilisateurs segmentés: {len(seg_usr):,}")

if risk is not None and not risk.empty:
    st.sidebar.write(f"⚠️ Utilisateurs évalués: {len(risk):,}")

# Data quality indicators
st.sidebar.markdown("---")
st.sidebar.subheader("Qualité des données")

if has_incidents:
    missing_sectors = (inc["Secteur"] == "Unknown").sum()
    if missing_sectors > 0:
        st.sidebar.warning(f"⚠️ {missing_sectors} incidents sans secteur")
    
    missing_impact = (inc["ImpactAriary"] == 0).sum()
    if missing_impact > 0:
        st.sidebar.info(f"ℹ️ {missing_impact} incidents sans impact financier")

if has_logins:
    unknown_results = (log["Resultat"] == "unknown").sum()
    if unknown_results > 0:
        st.sidebar.warning(f"⚠️ {unknown_results} connexions avec résultat inconnu")