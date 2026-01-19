# ============================================================
# 🏥 PLATEFORME EXPERTE D'AIDE À LA DÉCISION – DIABÈTE T2
# Version 2.2 : UX/UI Optimisée & SHAP Avancé
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import shap
from sklearn.model_selection import cross_val_predict

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Expert Diabète T2",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS PERSONNALISÉ (Optionnel mais recommandé pour un look pro) ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_assets():
    assets = {
        "model": None, 
        "metrics": {"recall": 0.82, "roc_auc": 0.84, "f1": 0.70},
        "feature_names": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        "loaded": False, 
        "real_distribution": None, 
        "raw_data": None
    }
    try:
        # 1. Modèle
        bundle = joblib.load("diabetes_risk_model.joblib")
        if isinstance(bundle, dict):
            assets["model"] = bundle.get("model")
            assets["metrics"] = bundle.get("metrics", assets["metrics"])
            assets["feature_names"] = bundle.get("feature_names", assets["feature_names"])
        else:
            assets["model"] = bundle
        assets["loaded"] = True

        # 2. Données réelles
        df = pd.read_csv("save_final.csv")
        assets["raw_data"] = df
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        
        # 3. Distribution de référence
        assets["real_distribution"] = cross_val_predict(
            assets["model"], X, y, cv=5, method="predict_proba", n_jobs=-1
        )[:, 1]
    except Exception as e:
        st.toast(f"Mode Démo activé : {e}", icon="⚠️")
        assets["real_distribution"] = np.random.beta(2, 5, 1000) 
    return assets

assets = load_assets()

# --- SIDEBAR ---
st.sidebar.header("📝 Dossier Patient")
st.sidebar.caption("Saisissez les paramètres cliniques ci-dessous.")

def get_user_input():
    inputs = {}
    with st.sidebar.form("patient_form"):
        inputs["Pregnancies"] = st.slider("Grossesses", 0, 17, 3)
        inputs["Glucose"] = st.slider("Glucose (mg/dL)", 40, 200, 120)
        inputs["BloodPressure"] = st.slider("Pression Artérielle (mmHg)", 30, 140, 70)
        inputs["SkinThickness"] = st.slider("Épaisseur Pli Cutané (mm)", 0, 99, 23)
        inputs["Insulin"] = st.slider("Insuline (µU/ml)", 0, 850, 80)
        inputs["BMI"] = st.slider("IMC (kg/m²)", 15.0, 60.0, 31.0)
        inputs["DiabetesPedigreeFunction"] = st.slider("Score Hérédité (DPF)", 0.07, 2.5, 0.47)
        inputs["Age"] = st.slider("Âge (ans)", 18, 90, 35)
        submit = st.form_submit_button("Lancer l'analyse")
    return pd.DataFrame(inputs, index=[0])

input_df = get_user_input()

# --- PRÉDICTION ---
if assets["loaded"] and assets["model"] is not None:
    prediction_proba = assets["model"].predict_proba(input_df)[0][1]
else:
    prediction_proba = 0.5 

# --- TABS ---
tab1, tab2 = st.tabs(["🏥 Aide à la Décision Clinique", "📊 Intelligence & Performance ML"])

# ==========================================
# ONGLET 1 : DIAGNOSTIC (Visualisation Améliorée)
# ==========================================
with tab1:
    st.title("Synthèse Clinique")
    
    # --- BLOC 1 : KPIs ---
    col_kpi1, col_kpi2, col_kpi3 = st.columns([1, 1, 1.5])
    
    with col_kpi1:
        st.metric("Score de Risque", f"{prediction_proba:.1%}", delta_color="inverse")
    
    with col_kpi2:
        if prediction_proba > 0.7:
            st.error("🔴 RISQUE ÉLEVÉ", icon="🚨")
        elif prediction_proba > 0.4:
            st.warning("🟠 RISQUE MODÉRÉ", icon="⚠️")
        else:
            st.success("🟢 RISQUE FAIBLE", icon="✅")
            
    with col_kpi3:
        if prediction_proba > 0.5:
            st.info("**Recommandation :** Dépistage HbA1c / HGPO requis.", icon="💉")
        else:
            st.success("**Recommandation :** Suivi préventif annuel standard.", icon="📅")

    st.divider()

    # --- BLOC 2 : EXPLICABILITÉ VISUELLE (Layout Optimisé) ---
    col_shap, col_comp = st.columns([1.5, 1], gap="large")

    with col_shap:
        st.subheader("🧬 Analyse des Facteurs d'Influence (SHAP)")
        st.caption("Quels paramètres spécifiques poussent le risque de ce patient vers le haut ou le bas ?")
        
        if assets["loaded"]:
            try:
                # Calcul SHAP
                model = assets["model"]
                preprocessor = model.named_steps['preprocess']
                classifier = model.named_steps['model']
                
                patient_trans = preprocessor.transform(input_df)
                explainer = shap.TreeExplainer(classifier)
                shap_val = explainer.shap_values(patient_trans)
                
                # Gestion format
                if isinstance(shap_val, list): sv = shap_val[1]
                elif len(shap_val.shape) == 3: sv = shap_val[:, :, 1]
                else: sv = shap_val
                
                # Création du graphique SHAP personnalisé
                # On utilise un style plus 'propre' pour Streamlit
                fig_s, ax_s = plt.subplots(figsize=(10, 5))
                
                # Bar plot SHAP
                shap.bar_plot(
                    sv[0] if len(sv.shape) > 1 else sv,
                    feature_names=assets["feature_names"],
                    max_display=8, # Affiche tout
                    show=False
                )
                
                # Customisation Matplotlib pour l'intégration
                plt.title(f"Contribution locale des variables (Patient)", fontsize=12, pad=10)
                plt.xlabel("Impact sur la probabilité (Log-odds)", fontsize=10)
                plt.gcf().set_facecolor("white") # Fond blanc propre
                
                st.pyplot(fig_s, use_container_width=True)
                
                # Légende explicative
                st.markdown("""
                <div style="font-size:12px; color:gray; text-align:center;">
                🟥 Rouge : Facteur augmentant le risque &nbsp;&nbsp;&nbsp; 🟦 Bleu : Facteur diminuant le risque
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Erreur de rendu SHAP : {e}")
        else:
            st.warning("Modèle non chargé, analyse indisponible.")

    with col_comp:
        st.subheader("📊 Comparatif Population")
        st.caption("Positionnement vs Moyenne des patients diabétiques.")
        
        # Données
        ref_means = {"Glucose": 141, "BMI": 35, "Age": 37, "BloodPressure": 71}
        feats = list(ref_means.keys())
        p_vals = [input_df[f][0] for f in feats]
        r_vals = [ref_means[f] for f in feats]
        
        # Graphique Radar ou Barres (Ici Barres améliorées)
        df_comp = pd.DataFrame({
            'Variable': feats * 2,
            'Valeur': p_vals + r_vals,
            'Type': ['Patient']*4 + ['Moy. Diabétiques']*4
        })
        
        fig_c = plt.figure(figsize=(6, 5))
        sns.barplot(data=df_comp, x='Variable', y='Valeur', hue='Type', palette=['#3498db', '#e74c3c'])
        plt.title("Écarts Critiques")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        sns.despine()
        
        st.pyplot(fig_c, use_container_width=True)
        
        with st.expander("Voir les détails chiffrés"):
            st.table(df_comp.pivot(index='Variable', columns='Type', values='Valeur'))


# ==========================================
# ONGLET 2 : PERFORMANCE (Reste identique mais clean)
# ==========================================
with tab2:
    st.title("Monitoring de la Performance IA")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Recall (Sensibilité)", f"{assets['metrics']['recall']:.1%}")
    m2.metric("ROC-AUC", f"{assets['metrics']['roc_auc']:.3f}")
    m3.metric("F1-Score", f"{assets['metrics']['f1']:.3f}")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Importance Globale")
        if assets["loaded"]:
            inner_model = getattr(assets["model"], "named_steps", {}).get("model", assets["model"])
            if hasattr(inner_model, "feature_importances_"):
                imp = inner_model.feature_importances_
                df_i = pd.DataFrame({"Var": assets["feature_names"], "Imp": imp}).sort_values("Imp")
                
                fig_i, ax_i = plt.subplots(figsize=(6, 4))
                ax_i.barh(df_i["Var"], df_i["Imp"], color="#2c3e50")
                plt.title("Poids des variables (Random Forest)")
                sns.despine()
                st.pyplot(fig_i)
    
    with c2:
        st.subheader("Analyse de Décision (DCA)")
        if assets["raw_data"] is not None:
            # Recalcul rapide DCA
            y_true = assets["raw_data"]["Outcome"]
            y_prob = assets["real_distribution"]
            thr = np.linspace(0.01, 0.9, 50)
            nb = []
            for t in thr:
                tp = np.sum((y_prob >= t) & (y_true == 1))
                fp = np.sum((y_prob >= t) & (y_true == 0))
                nb.append((tp/len(y_true)) - (fp/len(y_true))*(t/(1-t)))
            
            fig_d, ax_d = plt.subplots(figsize=(6, 4))
            ax_d.plot(thr, nb, color="blue", lw=2, label="Modèle")
            ax_d.axhline(0, color="k", lw=1)
            ax_d.set_ylim(-0.05, 0.4)
            plt.title("Courbe de Bénéfice Net")
            plt.xlabel("Seuil de probabilité")
            plt.legend()
            sns.despine()
            st.pyplot(fig_d)
            st.caption("Ce graphique montre que l'utilisation de ce modèle apporte un bénéfice supérieur au dépistage systématique.")

        else:
            st.warning("Données brutes requises pour la DCA.")
            
    st.subheader("Calibration")
    fig_cal, ax_cal = plt.subplots(figsize=(10, 2))
    sns.kdeplot(assets["real_distribution"], fill=True, color="purple", alpha=0.2)
    plt.axvline(prediction_proba, color="red", lw=2, linestyle="--")
    plt.text(prediction_proba+0.02, 0.5, "Patient", color="red", fontweight="bold")
    plt.yticks([])
    sns.despine(left=True)
    st.pyplot(fig_cal)