"""
=============================================================================
  LANGUAGE IDENTIFICATION CHALLENGE — Leaderboard App
  Streamlit application for automatic submission scoring
=============================================================================

  INSTALL :
    pip install streamlit pandas scikit-learn

  RUN :
    streamlit run leaderboard_app.py

  SETUP :
    Place secret_test_labels.csv in the same folder as this script
    OR set the path in SECRET_LABELS_PATH below.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path

try:
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# =============================================================================
#   CONFIGURATION — 
# =============================================================================

COMPETITION_TITLE   = "🌍 Language Identification Compétition"
#COMPETITION_SUBTITLE = "Multilingual LibriSpeech · 7 Languages · Deep Learning"
COMPETITION_SUBTITLE = "Team : Christian Munguagnaze Bwirachiza & Fatou Bintou Mbaye· Deep Learning Course @AIMS-Senegal 2026"
SECRET_LABELS_PATH  = "organizer/secret_test_labels.csv"   # chemin local
LEADERBOARD_FILE    = "leaderboard.json"                   # stockage persistant
MAX_SUBMISSIONS_DAY = 3
LANGUAGES           = ["fr", "de", "es", "it", "pt", "nl", "pl"]

LANG_NAMES = {
    "fr": "🇫🇷 French",   "de": "🇩🇪 German",  "es": "🇪🇸 Spanish",
    "it": "🇮🇹 Italian",  "pt": "🇵🇹 Portuguese", "nl": "🇳🇱 Dutch",
    "pl": "🇵🇱 Polish",
}


# =============================================================================
#   CHARGEMENT DES LABELS SECRETS
#   Priorité : 1) fichier local  2) Streamlit Secrets (pour déploiement cloud)
# =============================================================================

def load_secret_labels() -> pd.DataFrame:
    """
    Priorité 1 : Streamlit Secrets (fonctionne en local ET sur cloud)
    Priorité 2 : fichier local (fallback)
    """
    from io import StringIO

    # Priorité 1 — Streamlit Secrets (cloud ET local via .streamlit/secrets.toml)
    try:
        secret_csv = st.secrets["secret_labels_csv"]
        df = pd.read_csv(StringIO(secret_csv))
        cols = [c for c in ["clip_id", "language", "is_native", "accent_region"] if c in df.columns]
        if len(df) > 0:
            return df[cols]
    except Exception:
        pass

    # Priorité 2 — fichier local (fallback)
    possible_paths = [
        "organizer/secret_test_labels.csv",
        "Accent & language identification competition/organizer/secret_test_labels.csv",
        SECRET_LABELS_PATH,
    ]
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            cols = [c for c in ["clip_id", "language", "is_native", "accent_region"] if c in df.columns]
            return df[cols]

    return None


# =============================================================================
#   SCORING
# =============================================================================

def compute_score(submission: pd.DataFrame, secret: pd.DataFrame) -> dict:
    # Colonnes scoring uniquement — ignorer checksum, confidences etc.
    sub_cols    = ["clip_id", "language", "is_native"]
    secret_cols = ["clip_id", "language", "is_native"]
    sub    = submission[[c for c in sub_cols    if c in submission.columns]].copy()
    sec    = secret[[c    for c in secret_cols  if c in secret.columns]].copy()

    merged = sub.merge(sec, on="clip_id", suffixes=("_pred", "_true"))

    if len(merged) == 0:
        return {"error": "No matching clip_id found between submission and secret labels."}

    coverage = len(merged) / len(secret) * 100

    f1   = f1_score(merged["language_true"], merged["language_pred"],
                    average="weighted", zero_division=0)
    acc  = accuracy_score(merged["is_native_true"], merged["is_native_pred"])
    final = round(0.6 * f1 + 0.4 * acc, 4)

    # Per-language F1
    per_lang = {}
    for lang in LANGUAGES:
        mask = merged["language_true"] == lang
        if mask.sum() > 0:
            f1_lang = f1_score(
                merged.loc[mask, "language_true"],
                merged.loc[mask, "language_pred"],
                average="weighted", zero_division=0, labels=[lang]
            )
            per_lang[lang] = round(float(f1_lang), 4)

    return {
        "f1_language":  round(float(f1),   4),
        "acc_native":   round(float(acc),  4),
        "final_score":  final,
        "coverage":     round(coverage, 1),
        "n_predictions": len(merged),
        "per_language_f1": per_lang,
        "error": None,
    }


def validate_submission(df: pd.DataFrame) -> list:
    errors = []
    required = ["clip_id", "language", "is_native",
                "accent_region", "confidence_language", "confidence_native"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {', '.join(missing_cols)}")
        return errors

    invalid_langs = set(df["language"].unique()) - set(LANGUAGES)
    if invalid_langs:
        errors.append(f"Invalid language codes: {invalid_langs}. Expected: {set(LANGUAGES)}")

    if not df["is_native"].isin([0, 1]).all():
        errors.append("Column 'is_native' must contain only 0 or 1.")

    for col in ["confidence_language", "confidence_native"]:
        if col in df.columns:
            if not ((df[col] >= 0) & (df[col] <= 1)).all():
                errors.append(f"Column '{col}' must be in [0.0, 1.0].")

    if df["clip_id"].duplicated().any():
        errors.append("Duplicate clip_id values found.")

    return errors


# =============================================================================
#   LEADERBOARD PERSISTENCE
# =============================================================================

def load_leaderboard() -> list:
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    return []


def save_leaderboard(data: list):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2)


def count_today_submissions(leaderboard: list, team_name: str) -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    return sum(
        1 for e in leaderboard
        if e["team"] == team_name and e["timestamp"].startswith(today)
    )


def add_entry(leaderboard: list, team: str, scores: dict, filename: str) -> list:
    entry = {
        "team":         team,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "final_score":  scores["final_score"],
        "f1_language":  scores["f1_language"],
        "acc_native":   scores["acc_native"],
        "coverage":     scores["coverage"],
        "n_predictions": scores["n_predictions"],
        "per_language_f1": scores.get("per_language_f1", {}),
        "filename":     filename,
    }
    leaderboard.append(entry)
    save_leaderboard(leaderboard)
    return leaderboard


def get_best_per_team(leaderboard: list) -> pd.DataFrame:
    if not leaderboard:
        return pd.DataFrame()
    df = pd.DataFrame(leaderboard)
    best = df.groupby("team").apply(
        lambda x: x.loc[x["final_score"].idxmax()]
    ).reset_index(drop=True)
    best = best.sort_values("final_score", ascending=False).reset_index(drop=True)
    best.index += 1
    return best


# =============================================================================
#   STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Language ID Challenge",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.main { background-color: #0f1117; }

.medal-1 { color: #FFD700; font-size: 1.4em; }
.medal-2 { color: #C0C0C0; font-size: 1.4em; }
.medal-3 { color: #CD7F32; font-size: 1.4em; }

.score-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
}

.big-score {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    color: #48bb78;
    line-height: 1;
}

.score-label {
    font-size: 0.8rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}

.rank-badge {
    display: inline-block;
    background: #2d3748;
    color: #e2e8f0;
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.top-team {
    background: linear-gradient(135deg, #1a2f1a 0%, #0f1117 100%);
    border: 1px solid #276749;
}

.stButton > button {
    background: linear-gradient(135deg, #276749, #48bb78);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.6rem 2rem;
    width: 100%;
    font-size: 1rem;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.85; }

.warning-box {
    background: #2d2000;
    border: 1px solid #d69e2e;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #fbd38d;
    font-size: 0.9rem;
}

.success-box {
    background: #1a2f1a;
    border: 1px solid #48bb78;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #9ae6b4;
    font-size: 0.9rem;
}

.error-box {
    background: #2d1515;
    border: 1px solid #fc8181;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #feb2b2;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"# {COMPETITION_TITLE}")
st.markdown(f"*{COMPETITION_SUBTITLE}*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Competition Info")
    st.markdown(f"""
    | | |
    |---|---|
    | **Languages** | {len(LANGUAGES)} |
    | **Train clips** | 3,360 |
    | **Test clips** | 840 |
    | **Max submissions/day** | {MAX_SUBMISSIONS_DAY} |
    """)

    st.markdown("### 🎯 Scoring Formula")
    st.code("score = 0.6 × F1_lang\n      + 0.4 × Acc_native", language="text")

    st.markdown("### 🌍 Languages")
    for code, name in LANG_NAMES.items():
        st.markdown(f"- `{code}` {name}")

    st.markdown("### 📁 Submission Format")
    st.code("clip_id, language, is_native,\naccent_region,\nconfidence_language,\nconfidence_native", language="text")

    st.divider()
    st.markdown("### ⚙️ Admin")
    admin_pwd = st.text_input("Admin password", type="password")
    IS_ADMIN  = admin_pwd == st.secrets.get("admin_password", "admin123")
    if IS_ADMIN:
        st.success("✓ Admin mode active")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_lb, tab_submit, tab_analysis, tab_admin = st.tabs([
    "🏆 Leaderboard", "📤 Submit", "📊 Analysis", "⚙️ Admin"
])

leaderboard = load_leaderboard()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_lb:
    st.markdown("## 🏆 Leaderboard — Best Score per Team")

    best_df = get_best_per_team(leaderboard)

    if best_df.empty:
        st.info("No submissions yet. Be the first to submit!")
    else:
        # Top 3 cards
        top3 = best_df.head(3)
        medals = ["🥇", "🥈", "🥉"]
        cols = st.columns(min(3, len(top3)))

        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class="score-card {'top-team' if i == 0 else ''}">
                    <div class="score-label">{medals[i]} Rank {i+1}</div>
                    <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0; margin-bottom:0.5rem">
                        {row['team']}
                    </div>
                    <div class="big-score">{row['final_score']:.4f}</div>
                    <div style="margin-top:0.8rem; font-size:0.85rem; color:#718096">
                        F1 Lang: <b style="color:#90cdf4">{row['f1_language']:.4f}</b> &nbsp;|&nbsp;
                        Acc Native: <b style="color:#90cdf4">{row['acc_native']:.4f}</b>
                    </div>
                    <div style="font-size:0.75rem; color:#4a5568; margin-top:0.4rem">
                        {row['timestamp']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### 📋 Full Rankings")

        # Table
        display_df = best_df[["team", "final_score", "f1_language",
                               "acc_native", "coverage", "timestamp"]].copy()
        display_df.columns = ["Team", "Final Score", "F1 Language",
                               "Acc Native", "Coverage %", "Best Submission"]
        display_df.insert(
            0, "Rank",
            [medals[i] if i < 3 else f"#{i+1}" for i in range(len(display_df))]
        )
        display_df = display_df.reset_index(drop=True)
        
        score_min = display_df["Final Score"].min()
        score_max = display_df["Final Score"].max()
        
        if score_min == score_max:
            styled = display_df.style.format(
                {"Final Score": "{:.4f}", "F1 Language": "{:.4f}",
                 "Acc Native": "{:.4f}", "Coverage %": "{:.1f}%"}
            )
        else:
            styled = (
                display_df.style
                .background_gradient(
                    subset=["Final Score"], cmap="Greens",
                    vmin=score_min, vmax=score_max
                )
                .format({"Final Score": "{:.4f}", "F1 Language": "{:.4f}",
                         "Acc Native": "{:.4f}", "Coverage %": "{:.1f}%"})
            )

        # Refresh button
        if st.button("🔄 Refresh Leaderboard"):
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SUBMIT
# ══════════════════════════════════════════════════════════════════════════════
with tab_submit:
    st.markdown("## 📤 Submit Your Predictions")

    if not SKLEARN_OK:
        st.error("scikit-learn not installed. Run: `pip install scikit-learn`")
        st.stop()

    secret_df = load_secret_labels()
    if secret_df is None:
        st.error("❌ Secret labels not configured. Submissions cannot be scored.")
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 💻 En local")
            st.markdown("Copiez le fichier généré par le builder dans le bon dossier :")
            st.code("mkdir organizer\ncp accent_dataset_output/organizer/secret_test_labels.csv ./organizer/", language="bash")
        with col_b:
            st.markdown("#### ☁️ Sur Streamlit Cloud")
            st.markdown("Dans **Settings → Secrets**, ajoutez le contenu du fichier :")
            st.code('secret_labels_csv = """\nclip_id,language,is_native,accent_region\nfr_000001,fr,1,paris\nde_000001,de,0,turkish\n...\n"""', language="toml")
            st.info("Copiez-collez TOUT le contenu de `secret_test_labels.csv` entre les guillemets.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        team_name = st.text_input(
            "👤 Team / Participant name",
            placeholder="e.g. Team Me & You",
            help="Your name or team name as it will appear on the leaderboard"
        )

    with col2:
        uploaded = st.file_uploader(
            "📁 Upload submission.csv",
            type=["csv"],
            help="Must follow the submission format shown in the sidebar"
        )

    if team_name and uploaded:
        # Count today's submissions
        n_today = count_today_submissions(leaderboard, team_name)
        remaining = MAX_SUBMISSIONS_DAY - n_today

        if remaining <= 0:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ <b>{team_name}</b> has reached the daily limit
                ({MAX_SUBMISSIONS_DAY} submissions/day).
                Come back tomorrow!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"📊 Submissions today: **{n_today}/{MAX_SUBMISSIONS_DAY}** — {remaining} remaining")

            if st.button("🚀 Score My Submission"):
                with st.spinner("Scoring your submission..."):
                    try:
                        sub_df = pd.read_csv(uploaded)
                        # secret_df already loaded above

                        # Validate
                        errors = validate_submission(sub_df)
                        if errors:
                            st.markdown(f"""
                            <div class="error-box">
                                <b>❌ Submission format errors:</b><br>
                                {"<br>".join(f"• {e}" for e in errors)}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            scores = compute_score(sub_df, secret_df)

                            if scores.get("error"):
                                st.error(scores["error"])
                            else:
                                # Add to leaderboard
                                leaderboard = add_entry(
                                    leaderboard, team_name, scores, uploaded.name
                                )

                                # Show results
                                st.markdown(f"""
                                <div class="success-box">
                                    ✅ <b>Submission scored successfully!</b>
                                </div>
                                """, unsafe_allow_html=True)

                                st.markdown("### 🎯 Your Scores")
                                c1, c2, c3 = st.columns(3)

                                with c1:
                                    st.markdown(f"""
                                    <div class="score-card">
                                        <div class="score-label">🏆 Final Score</div>
                                        <div class="big-score">{scores['final_score']:.4f}</div>
                                        <div style="color:#718096;font-size:0.8rem">0.6×F1 + 0.4×Acc</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                with c2:
                                    st.markdown(f"""
                                    <div class="score-card">
                                        <div class="score-label">🌍 F1 Language</div>
                                        <div class="big-score" style="color:#90cdf4">{scores['f1_language']:.4f}</div>
                                        <div style="color:#718096;font-size:0.8rem">weighted F1-score</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                with c3:
                                    st.markdown(f"""
                                    <div class="score-card">
                                        <div class="score-label">🎙 Acc Native</div>
                                        <div class="big-score" style="color:#fbd38d">{scores['acc_native']:.4f}</div>
                                        <div style="color:#718096;font-size:0.8rem">binary accuracy</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Rank on leaderboard
                                best_df_new = get_best_per_team(leaderboard)
                                rank = best_df_new[best_df_new["team"] == team_name].index
                                if len(rank) > 0:
                                    r = rank[0]
                                    medal = ["🥇","🥈","🥉"][r-1] if r <= 3 else f"#{r}"
                                    st.success(f"Your current rank: **{medal} Position {r}** out of {len(best_df_new)} teams")

                                # Per-language breakdown
                                if scores.get("per_language_f1"):
                                    st.markdown("### 🌍 Per-Language F1 Breakdown")
                                    lang_data = {
                                        LANG_NAMES.get(k, k): v
                                        for k, v in scores["per_language_f1"].items()
                                    }
                                    lang_df = pd.DataFrame(
                                        list(lang_data.items()),
                                        columns=["Language", "F1 Score"]
                                    ).sort_values("F1 Score", ascending=False)

                                    st.bar_chart(lang_df.set_index("Language"))

                    except Exception as e:
                        st.error(f"Error processing submission: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.markdown("## 📊 Competition Analysis")

    if not leaderboard:
        st.info("No submissions yet.")
    else:
        df_all = pd.DataFrame(leaderboard)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Score Distribution")
            st.bar_chart(df_all.groupby("team")["final_score"].max().sort_values(ascending=False))

        with col2:
            st.markdown("### 📅 Submissions Over Time")
            daily = df_all.groupby(df_all["timestamp"].dt.date).size()
            st.bar_chart(daily)

        st.markdown("### 📋 All Submissions History")
        history = df_all[["team", "timestamp", "final_score",
                           "f1_language", "acc_native", "filename"]].copy()
        history = history.sort_values("timestamp", ascending=False)
        history["timestamp"] = history["timestamp"].astype(str)
        st.dataframe(history, use_container_width=True)

        st.markdown("### 📊 Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Submissions", len(df_all))
        c2.metric("Teams", df_all["team"].nunique())
        c3.metric("Best Score", f"{df_all['final_score'].max():.4f}")
        c4.metric("Average Score", f"{df_all['final_score'].mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ADMIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    st.markdown("## ⚙️ Admin Panel")

    if not IS_ADMIN:
        st.warning("Enter the admin password in the sidebar to access this panel.")
    else:
        st.success("✓ Admin access granted")

        st.markdown("### 🗑️ Reset Leaderboard")
        if st.button("⚠️ Clear all submissions", type="primary"):
            save_leaderboard([])
            st.success("Leaderboard cleared.")
            st.rerun()

        st.markdown("### 🚫 Remove a Team")
        if leaderboard:
            teams = list(set(e["team"] for e in leaderboard))
            team_to_remove = st.selectbox("Select team to remove", teams)
            if st.button(f"Remove {team_to_remove}"):
                leaderboard = [e for e in leaderboard if e["team"] != team_to_remove]
                save_leaderboard(leaderboard)
                st.success(f"Team '{team_to_remove}' removed.")
                st.rerun()

        st.markdown("### 📥 Export Leaderboard")
        if leaderboard:
            best = get_best_per_team(leaderboard)
            csv = best.to_csv(index=True)
            st.download_button(
                "⬇️ Download leaderboard.csv",
                data=csv,
                file_name="final_leaderboard.csv",
                mime="text/csv"
            )

        st.markdown("### 🔑 Change Admin Password")
        st.info("Set `admin_password` in `.streamlit/secrets.toml`:\n```toml\nadmin_password = 'your_password'\n```")
