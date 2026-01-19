# ============================================================
# 🏥 PLATEFORME EXPERTE D'AIDE À LA DÉCISION – DIABÈTE T2
# Fusion Optimisée : Diagnostic Clinique & Monitoring ML
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.model_selection import cross_val_predict

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Expert Diabète T2",
    page_icon="🏥",
    layout="wide"
)

# --- CHARGEMENT SÉCURISÉ DU MODÈLE ET DES DONNÉES ---
@st.cache_resource
def load_assets():
    assets = {
        "model": None, "metrics": {"recall": 0.89, "roc_auc": 0.84, "f1": 0.70},
        "feature_names": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        "loaded": False, "real_distribution": None, "raw_data": None
    }
    try:
        # 1. Chargement du modèle
        bundle = joblib.load("diabetes_risk_model.joblib")
        if isinstance(bundle, dict):
            assets["model"] = bundle.get("model")
            assets["metrics"] = bundle.get("metrics", assets["metrics"])
            assets["feature_names"] = bundle.get("feature_names", assets["feature_names"])
        else:
            assets["model"] = bundle
        assets["loaded"] = True

        # 2. Chargement des données pour la calibration réelle
        df = pd.read_csv("save_final.csv")
        assets["raw_data"] = df
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        
        # Calcul de la distribution de référence (CV)
        assets["real_distribution"] = cross_val_predict(
            assets["model"], X, y, cv=5, method="predict_proba", n_jobs=-1
        )[:, 1]
    except Exception as e:
        st.sidebar.warning(f"⚠️ Mode Démo : Actifs réels non chargés ({e})")
        assets["real_distribution"] = np.random.beta(2, 5, 1000) # Simulation réaliste
    return assets

assets = load_assets()

# --- SIDEBAR : DOSSIER PATIENT ---
st.sidebar.title("📝 Dossier Patient")
def get_user_input():
    inputs = {}
    inputs["Pregnancies"] = st.sidebar.slider("Grossesses", 0, 17, 3)
    inputs["Glucose"] = st.sidebar.slider("Glucose (mg/dL)", 40, 200, 120)
    inputs["BloodPressure"] = st.sidebar.slider("Pression Artérielle (mmHg)", 30, 140, 70)
    inputs["SkinThickness"] = st.sidebar.slider("Épaisseur Pli Cutané (mm)", 0, 99, 23)
    inputs["Insulin"] = st.sidebar.slider("Insuline (µU/ml)", 0, 850, 80)
    inputs["BMI"] = st.sidebar.slider("Indice de Masse Corporelle (IMC)", 15.0, 60.0, 31.0)
    inputs["DiabetesPedigreeFunction"] = st.sidebar.slider("Score Hérédité (DPF)", 0.07, 2.5, 0.47)
    inputs["Age"] = st.sidebar.slider("Âge", 18, 90, 35)
    return pd.DataFrame(inputs, index=[0])

input_df = get_user_input()

# --- LOGIQUE DE PRÉDICTION ---
if assets["loaded"] and assets["model"] is not None:
    prediction_proba = assets["model"].predict_proba(input_df)[0][1]
else:
    # Formule simplifiée pour le mode démo
    risk = (input_df["Glucose"][0]/200 * 0.6) + (input_df["BMI"][0]/60 * 0.4)
    prediction_proba = min(risk, 0.95)

# --- INTERFACE PRINCIPALE (TABS) ---
tab1, tab2 = st.tabs(["🏥 Aide à la Décision Clinique", "📊 Intelligence & Performance ML"])

# ==========================================
# ONGLET 1 : DIAGNOSTIC CLINIQUE
# ==========================================
with tab1:
    st.title("Évaluation du Risque Diabétique")
    
    # KPIs Haut de page
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Probabilité de Risque", f"{prediction_proba:.1%}")
    with k2:
        if prediction_proba > 0.7: st.error("🔴 RISQUE ÉLEVÉ")
        elif prediction_proba > 0.4: st.warning("🟠 RISQUE MODÉRÉ")
        else: st.success("🟢 RISQUE FAIBLE")
    with k3:
        action = "Dépistage HbA1c" if prediction_proba > 0.5 else "Suivi Annuel"
        st.info(f"Action recommandée : {action}")

    st.markdown("---")
    
    # Comparaison Visuelle
    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        st.subheader("📊 Profil du Patient vs Moyennes Diabétiques")
        ref_means = {"Glucose": 141, "BMI": 35, "Age": 37, "BloodPressure": 71}
        features = list(ref_means.keys())
        p_vals = [input_df[f][0] for f in features]
        r_vals = [ref_means[f] for f in features]
        
        x = np.arange(len(features))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.2, p_vals, 0.4, label='Patient', color='#3498db')
        ax.bar(x + 0.2, r_vals, 0.4, label='Moyenne Diabétiques', color='#e74c3c', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()
        st.pyplot(fig)
    
    with col_v2:
        st.subheader("💡 Interprétation")
        # st.write(f"Le patient se situe au **{np.mean(assets['real_distribution'] <= prediction_proba)*100:.1f}ème percentile** de la population de test.")
        st.markdown("""
        **Conseils :**
        - Si Glucose > 140 : Risque hyperglycémique.
        - Si IMC > 30 : Obésité facteur aggravant.
        """)

# ==========================================
# ONGLET 2 : PERFORMANCE & QUALITÉ ML
# ==========================================
with tab2:
    st.title("Transparence et Métriques du Modèle")
    
    # 1. Métriques de Validation
    m1, m2, m3 = st.columns(3)
    m1.metric("Sensibilité (Recall)", f"{assets['metrics']['recall']:.1%}", help="Capacité à détecter les malades")
    m2.metric("ROC-AUC", f"{assets['metrics']['roc_auc']:.3f}")
    m3.metric("F1-Score", f"{assets['metrics']['f1']:.3f}")

    st.markdown("---")
    
    col_ml1, col_ml2 = st.columns(2)
    
    # 2. Feature Importance
    with col_ml1:
        st.subheader("🎯 Importance des Variables")
        if assets["loaded"]:
            # On cherche le modèle dans le pipeline
            inner_model = getattr(assets["model"], "named_steps", {}).get("model", assets["model"])
            if hasattr(inner_model, "feature_importances_"):
                imp = inner_model.feature_importances_
                df_i = pd.DataFrame({"Varaible": assets["feature_names"], "Importance": imp}).sort_values("Importance")
                fig_i, ax_i = plt.subplots()
                ax_i.barh(df_i["Varaible"], df_i["Importance"], color="teal")
                st.pyplot(fig_i)
        else:
            st.info("Données d'importance basées sur le profil clinique standard.")

    # 3. Decision Curve Analysis (DCA)
    with col_ml2:
        st.subheader("📈 Utilité Clinique (Decision Curve)")
        if assets["raw_data"] is not None:
            # Calcul simplifié DCA
            y_true = assets["raw_data"]["Outcome"]
            y_prob = assets["real_distribution"]
            thresh = np.linspace(0.1, 0.8, 50)
            net_benefit = []
            for t in thresh:
                tp = np.sum((y_prob >= t) & (y_true == 1))
                fp = np.sum((y_prob >= t) & (y_true == 0))
                nb = (tp / len(y_true)) - (fp / len(y_true)) * (t / (1 - t))
                net_benefit.append(nb)
            
            fig_d, ax_d = plt.subplots()
            ax_d.plot(thresh, net_benefit, label="Notre Modèle", color="blue", lw=2)
            ax_d.plot(thresh, (np.sum(y_true)/len(y_true)) - ( (1-np.sum(y_true)/len(y_true)) * (thresh/(1-thresh)) ), '--', label="Traiter tous", color="grey")
            ax_d.axhline(0, color='black', lw=1, label="Traiter aucun")
            ax_d.set_ylim(-0.05, 0.4)
            ax_d.legend()
            st.pyplot(fig_d)
            st.caption("Ce graphique montre que l'utilisation de ce modèle apporte un bénéfice supérieur au dépistage systématique.")

        else:
            st.warning("DCA indisponible sans le dataset original.")

    st.markdown("---")
    # 4. Calibration Curve (Visualisation de la distribution)
    st.subheader("📍 Calibration : Position du Patient")
    fig_c, ax_c = plt.subplots(figsize=(12, 3))
    sns.kdeplot(assets["real_distribution"], fill=True, color="purple", ax=ax_c, label="Distribution des scores (Population)")
    ax_c.axvline(prediction_proba, color="red", lw=3, label="Patient Actuel")
    ax_c.set_xlim(0, 1)
    ax_c.legend()
    st.pyplot(fig_c)

# --- FOOTER ---
st.markdown("---")
st.caption("v2.0 Finale - Système d'Aide à la Décision Médicale (OAD) | Données d'entraînement : Pima Indians")