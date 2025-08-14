# m8_dashboard.py
# Lance avec: streamlit run m8_dashboard.py
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="ESN Cybersécurité — Dashboard", layout="wide")

@st.cache_data
def load_data():
    inc = pd.read_csv("incidents.csv", parse_dates=["Date"])
    log = pd.read_csv("logins.csv", parse_dates=["DateHeure"])
    seg_ent = pd.read_csv("segments_entreprises.csv") if "segments_entreprises.csv" else None
    seg_usr = pd.read_csv("segments_utilisateurs.csv") if "segments_utilisateurs.csv" else None
    risk = pd.read_csv("risque_utilisateur.csv") if "risque_utilisateur.csv" else None
    return inc, log, seg_ent, seg_usr, risk

inc, log, seg_ent, seg_usr, risk = load_data()

st.title("IAro Tech — Dashboard")

# Filtres
secteurs = ["(Tous)"] + sorted(inc["Secteur"].dropna().unique().tolist())
secteur_sel = st.sidebar.selectbox("Secteur", secteurs)
type_att_sel = st.sidebar.multiselect("Types d’attaque", sorted(inc["TypeAttaque"].dropna().unique().tolist()))
role_sel = st.sidebar.multiselect("Rôles", sorted(log["Role"].dropna().unique().tolist()))

df_inc = inc.copy()
if secteur_sel != "(Tous)":
    df_inc = df_inc[df_inc["Secteur"]==secteur_sel]
if type_att_sel:
    df_inc = df_inc[df_inc["TypeAttaque"].isin(type_att_sel)]

df_log = log.copy()
if role_sel:
    df_log = df_log[df_log["Role"].isin(role_sel)]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Incidents (filtrés)", len(df_inc))
col2.metric("Impact total (MGA)", f"{df_inc['ImpactAriary'].sum():,.0f}")
col3.metric("Taux d’échec (logins)", f"{(df_log['Resultat'].eq('échec').mean() if len(df_log)>0 else 0):.2%}")

# Graph 1 : Incidents / mois
st.subheader("Incidents par mois")
inc_m = df_inc.set_index("Date").resample("M").size()
fig1, ax1 = plt.subplots()
inc_m.plot(ax=ax1)
ax1.set_title("Incidents par mois")
ax1.set_xlabel("Mois")
ax1.set_ylabel("Incidents")
st.pyplot(fig1)

# Graph 2 : Impact € par secteur (médiane)
st.subheader("Impact médian par secteur (MGA)")
imp_sec = df_inc.groupby("Secteur")["ImpactAriary"].median().sort_values(ascending=False)
fig2, ax2 = plt.subplots()
imp_sec.plot(kind="bar", ax=ax2)
ax2.set_title("Impact médian par secteur")
ax2.set_xlabel("Secteur")
ax2.set_ylabel("MGA médian")
st.pyplot(fig2)

# Graph 3 : Taux d’échec des connexions (mensuel)
st.subheader("Taux d’échec des connexions (mensuel)")
log_m = df_log.set_index("DateHeure").resample("M")["Resultat"].apply(lambda s: (s=="échec").mean())
fig3, ax3 = plt.subplots()
log_m.plot(ax=ax3)
ax3.set_title("Taux d’échec (mensuel)")
ax3.set_xlabel("Mois")
ax3.set_ylabel("Taux")
st.pyplot(fig3)

# Tables segments / risques
st.subheader("Segments entreprises")
if seg_ent is not None:
    st.dataframe(seg_ent.sort_values("cluster_esn"))

st.subheader("Segments utilisateurs & risque")
if seg_usr is not None:
    st.dataframe(seg_usr.sort_values("cluster_risque"))
if risk is not None:
    st.dataframe(risk.sort_values("risk_score", ascending=False).head(50))
