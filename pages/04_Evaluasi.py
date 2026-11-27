from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components.empty_state import tampilkan_empty_state
from utils.artifact_loader import (
    GAMBAR,
    muat_feature_importance,
    muat_hasil_evaluasi,
    muat_kfold_raw,
    muat_ranking_shap,
    muat_shap_global,
    path_gambar,
)


ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT / "assets" / "styles.css"
st.set_page_config(page_title="Evaluasi & Interpretasi", page_icon="📊", layout="wide")
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


METRIC_LABEL = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
}


def format_metric_table(holdout: dict) -> pd.DataFrame:
    rows = []
    for model, values in holdout.items():
        rows.append({
            "Model": model,
            "Accuracy": values.get("accuracy"),
            "Precision": values.get("precision"),
            "Recall": values.get("recall"),
            "F1-Score": values.get("f1_score"),
        })
    return pd.DataFrame(rows)


def format_cv_table(kfold: list | None, raw: pd.DataFrame | None) -> pd.DataFrame:
    if kfold:
        rows = []
        for item in kfold:
            rows.append({
                "Model": item.get("model"),
                "Accuracy": f"{item.get('accuracy_mean', float('nan')):.2%} ± {item.get('accuracy_std', float('nan')):.2%}",
                "Precision": f"{item.get('precision_mean', float('nan')):.2%} ± {item.get('precision_std', float('nan')):.2%}",
                "Recall": f"{item.get('recall_mean', float('nan')):.2%} ± {item.get('recall_std', float('nan')):.2%}",
                "F1-Score": f"{item.get('f1_score_mean', float('nan')):.2%} ± {item.get('f1_score_std', float('nan')):.2%}",
            })
        return pd.DataFrame(rows)

    if raw is not None:
        summary = raw.groupby("model")[["accuracy", "precision", "recall", "f1_score"]].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        rows = []
        for _, item in summary.iterrows():
            rows.append({
                "Model": item["model"],
                "Accuracy": f"{item['accuracy_mean']:.2%} ± {item['accuracy_std']:.2%}",
                "Precision": f"{item['precision_mean']:.2%} ± {item['precision_std']:.2%}",
                "Recall": f"{item['recall_mean']:.2%} ± {item['recall_std']:.2%}",
                "F1-Score": f"{item['f1_score_mean']:.2%} ± {item['f1_score_std']:.2%}",
            })
        return pd.DataFrame(rows)
    return pd.DataFrame()


def plot_confusion(values: dict, title: str):
    matrix = [[values["TN"], values["FP"]], [values["FN"], values["TP"]]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    colors = [["#355f51", "#f3ddd5"], ["#f4d58d", "#355f51"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.96, 0.96, color=colors[i][j]))
            text_color = "white" if colors[i][j] == "#355f51" else "#20302b"
            ax.text(j + 0.48, 1 - i + 0.58, str(matrix[i][j]), ha="center", va="center",
                    fontsize=21, fontweight="bold", color=text_color)
            ax.text(j + 0.48, 1 - i + 0.25, labels[i][j], ha="center", va="center",
                    fontsize=10, color=text_color)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.48, 1.48], ["Prediksi Benign", "Prediksi Malignant"])
    ax.set_yticks([1.48, 0.48], ["Aktual Benign", "Aktual Malignant"])
    ax.set_title(title)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


st.title("Evaluasi & Interpretabilitas Model")
st.caption("Sumber visualisasi: artefak hasil training pada folder models/ dan gambar/. Tidak ada training atau SHAP yang dijalankan oleh halaman ini.")

hasil = muat_hasil_evaluasi()
raw_kfold, kfold_reason = muat_kfold_raw()
fi, fi_reason = muat_feature_importance()
shap_ranking, shap_reason = muat_ranking_shap()
shap_global, shap_global_reason = muat_shap_global()

tab_performa, tab_cm, tab_fi, tab_shap = st.tabs([
    "Performa Model",
    "Confusion Matrix",
    "Feature Importance",
    "SHAP Explainability",
])

with tab_performa:
    if not hasil["available"]:
        tampilkan_empty_state("Hasil evaluasi belum tersedia", hasil["reason"], icon="📊")
    else:
        st.markdown("### Holdout 80:20")
        holdout_table = format_metric_table(hasil["holdout"])
        if holdout_table.empty:
            tampilkan_empty_state("Holdout belum tersedia", "models/hasil_evaluasi.json tidak memuat hasil holdout.", icon="📊")
        else:
            st.dataframe(
                holdout_table.style.format({
                    "Accuracy": "{:.2%}", "Precision": "{:.2%}",
                    "Recall": "{:.2%}", "F1-Score": "{:.2%}",
                }),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("### Repeated Stratified 10-Fold × 5")
        cv_table = format_cv_table(hasil.get("kfold"), raw_kfold)
        if cv_table.empty:
            tampilkan_empty_state("Hasil cross-validation belum tersedia", kfold_reason or "Ringkasan k-fold tidak tersedia.", icon="🔁")
        else:
            st.dataframe(cv_table, hide_index=True, use_container_width=True)

        st.caption("Kelas positif: Malignant (label 1). Holdout dan cross-validation ditampilkan terpisah.")

        with st.expander("Penjelasan metrik"):
            st.markdown("**Accuracy:** proporsi seluruh prediksi yang benar.")
            st.markdown("**Precision:** proporsi prediksi Malignant yang benar.")
            st.markdown("**Recall:** proporsi pasien Malignant aktual yang terdeteksi.")
            st.markdown("**F1-Score:** rata-rata harmonik Precision dan Recall.")

with tab_cm:
    if not hasil["available"]:
        tampilkan_empty_state("Confusion matrix belum tersedia", hasil["reason"], icon="▦")
    else:
        holdout = hasil["holdout"]
        available_models = [model for model in ("Random Forest", "SVM RBF") if model in holdout]
        if not available_models:
            tampilkan_empty_state("Confusion matrix belum tersedia", "Tidak ada TP, TN, FP, dan FN pada artefak holdout.", icon="▦")
        else:
            columns = st.columns(len(available_models))
            for col, model in zip(columns, available_models):
                with col:
                    values = holdout[model]
                    if all(key in values for key in ("TN", "FP", "FN", "TP")):
                        fig = plot_confusion(values, f"{model} — Holdout")
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        st.caption(
                            f"TN={values['TN']} | FP={values['FP']} | "
                            f"FN={values['FN']} | TP={values['TP']}"
                        )
                    else:
                        tampilkan_empty_state(
                            f"{model}: data tidak lengkap",
                            "Artefak holdout tidak memuat empat sel confusion matrix.",
                            icon="▦",
                        )

            st.caption("Orientasi matriks mengikuti evaluasi asli: baris aktual, kolom prediksi; baris atas Benign, baris bawah Malignant.")

with tab_fi:
    if fi is None:
        tampilkan_empty_state("Feature Importance belum tersedia", fi_reason, icon="🌲")
    else:
        st.markdown("### Top 10 fitur — Random Forest")
        top10 = fi.sort_values("importance", ascending=False).head(10).iloc[::-1]
        st.bar_chart(top10.set_index("nama")["importance"])
        st.caption("Nilai dibaca dari models/feature_importance_rf.csv; bukan dihitung ulang oleh UI.")
        st.markdown("### Ranking seluruh fitur")
        st.dataframe(
            fi.style.format({"importance": "{:.6f}"}),
            hide_index=True,
            use_container_width=True,
            height=520,
        )

        png_fi = path_gambar("feature_importance_rf.png")
        if png_fi:
            st.image(str(png_fi), caption="Artefak grafik Feature Importance hasil training")

with tab_shap:
    if shap_ranking is None:
        tampilkan_empty_state("SHAP belum tersedia", shap_reason, icon="🔬")
    else:
        st.markdown("### Global feature importance — SVM RBF")
        top10 = shap_ranking.sort_values("mean_abs_shap", ascending=False).head(10).iloc[::-1]
        st.bar_chart(top10.set_index("nama")["mean_abs_shap"])
        st.caption("Nilai adalah mean(|SHAP|) dari artefak ranking_shap_svm.csv.")

        st.markdown("### Ranking seluruh fitur")
        st.dataframe(
            shap_ranking.style.format({"mean_abs_shap": "{:.6f}"}),
            hide_index=True,
            use_container_width=True,
            height=520,
        )

        png_summary = path_gambar("shap_summary_svm.png")
        png_global = path_gambar("shap_global_svm.png")
        png_local = path_gambar("shap_lokal_contoh.png")

        if png_summary:
            st.image(str(png_summary), caption="Summary plot SVM RBF dari artefak training")
        if png_global:
            st.image(str(png_global), caption="Global SHAP bar chart dari artefak training")

        if png_local:
            st.markdown("### Local explanation")
            st.image(str(png_local), caption="Contoh SHAP lokal satu pasien Malignant pada data uji")
        else:
            tampilkan_empty_state(
                "Local explanation belum tersedia",
                "gambar/shap_lokal_contoh.png belum tersedia.",
                icon="📍",
            )

        if shap_global is not None:
            st.caption(
                "SHAP menjelaskan probabilitas Malignant pada model SVM. "
                "Nilai positif mendorong ke Malignant; nilai negatif mendorong ke Benign."
            )
        else:
            st.caption(shap_global_reason)

st.markdown('<div class="disclaimer">', unsafe_allow_html=True)
st.markdown(
    "Hasil hanya boleh ditafsirkan sebagai hasil eksperimen yang benar-benar tersimpan. "
    "Sumber file dan metode evaluasi ditampilkan pada setiap section."
)
st.markdown('</div>', unsafe_allow_html=True)
