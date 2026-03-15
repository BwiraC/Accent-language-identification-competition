"""
=============================================================================
  LANGUAGE IDENTIFICATION CHALLENGE — Leaderboard App  (v3 — Google Sheets)
=============================================================================

  INSTALL :
    pip install streamlit pandas scikit-learn gspread google-auth

  RUN :
    streamlit run leaderboard_app.py

  PERSISTENCE :
    Toutes les soumissions sont stockées dans Google Sheets.
    Fonctionne identiquement en LOCAL et sur STREAMLIT CLOUD.
    Les données ne sont jamais perdues au redémarrage.

  SETUP (à faire une seule fois) :
    Voir le fichier SETUP_GOOGLE_SHEETS.md livré avec ce script.

=============================================================================
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from io import StringIO

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_OK = True
except ImportError:
    GSPREAD_OK = False

try:
    from sklearn.metrics import f1_score, accuracy_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# =============================================================================
#   CONFIGURATION
# =============================================================================

COMPETITION_TITLE    = "🌍 Language Identification Compétition"
COMPETITION_SUBTITLE = "Team : Christian Munguagnaze Bwirachiza & Fatou Bintou Mbaye · Deep Learning Course @AIMS-Senegal 2026"
MAX_SUBMISSIONS_DAY  = 3
LANGUAGES            = ["fr", "de", "es", "it", "pt", "nl", "pl"]

LANG_NAMES = {
    "fr": "🇫🇷 French",   "de": "🇩🇪 German",  "es": "🇪🇸 Spanish",
    "it": "🇮🇹 Italian",  "pt": "🇵🇹 Portuguese", "nl": "🇳🇱 Dutch",
    "pl": "🇵🇱 Polish",
}

SHEET_LEADERBOARD = "leaderboard"
SHEET_SUBMISSIONS = "submissions"

# =============================================================================
#   CONNEXION GOOGLE SHEETS
# =============================================================================

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource(show_spinner="Connecting to Google Sheets…")
def get_gsheet_client():
    if not GSPREAD_OK:
        return None
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"❌ Google Sheets connection failed: {e}")
        return None


@st.cache_resource(show_spinner="Opening spreadsheet…")
def get_spreadsheet():
    client = get_gsheet_client()
    if client is None:
        return None
    try:
        url = st.secrets["spreadsheet_url"]
        return client.open_by_url(url)
    except Exception as e:
        st.error(f"❌ Cannot open spreadsheet: {e}")
        return None


def get_or_create_worksheet(spreadsheet, name: str, headers: list):
    try:
        ws = spreadsheet.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=name, rows=2000, cols=len(headers))
        ws.append_row(headers)
    return ws

# =============================================================================
#   LEADERBOARD — lecture / écriture Google Sheets
# =============================================================================

LEADERBOARD_HEADERS = [
    "team", "timestamp", "final_score", "f1_language", "acc_native",
    "coverage", "n_predictions", "per_language_f1", "filename"
]
SUBMISSION_HEADERS = ["team", "timestamp", "filename", "csv_content"]


def load_leaderboard() -> list:
    sp = get_spreadsheet()
    if sp is None:
        return []
    try:
        ws = get_or_create_worksheet(sp, SHEET_LEADERBOARD, LEADERBOARD_HEADERS)
        records = ws.get_all_records()
        for r in records:
            if isinstance(r.get("per_language_f1"), str):
                try:
                    r["per_language_f1"] = json.loads(r["per_language_f1"])
                except Exception:
                    r["per_language_f1"] = {}

            r["final_score"]   = round(float(r.get("final_score",   0)), 4)
            r["f1_language"]   = round(float(r.get("f1_language",   0)), 4)
            r["acc_native"]    = round(float(r.get("acc_native",    0)), 4)
            r["coverage"]      = float(r.get("coverage",      0))
            r["n_predictions"] = int(r.get("n_predictions",   0))

            for key in ["final_score", "f1_language", "acc_native"]:
              if r[key] > 1:
                r[key] = round(r[key] / 10000, 4)
        return records
    except Exception as e:
        st.error(f"Error loading leaderboard: {e}")
        return []


def append_leaderboard_entry(entry: dict):
    sp = get_spreadsheet()
    if sp is None:
        return
    try:
        ws = get_or_create_worksheet(sp, SHEET_LEADERBOARD, LEADERBOARD_HEADERS)
        row = [
            entry["team"],
            entry["timestamp"],
            entry["final_score"],
            entry["f1_language"],
            entry["acc_native"],
            entry["coverage"],
            entry["n_predictions"],
            json.dumps(entry.get("per_language_f1", {})),
            entry["filename"],
        ]
        #ws.append_row(row, value_input_option="USER_ENTERED")
        ws.append_row(row, value_input_option="RAW")
    except Exception as e:
        st.error(f"Error saving entry: {e}")


def save_submission_csv(team: str, timestamp: str, filename: str, df: pd.DataFrame):
    sp = get_spreadsheet()
    if sp is None:
        return
    try:
        ws = get_or_create_worksheet(sp, SHEET_SUBMISSIONS, SUBMISSION_HEADERS)
        ws.append_row([team, timestamp, filename, df.to_csv(index=False)],
                      value_input_option="USER_ENTERED")
    except Exception:
        pass  # non-bloquant


def delete_team_entries(team_name: str):
    sp = get_spreadsheet()
    if sp is None:
        return
    for sheet_name in [SHEET_LEADERBOARD, SHEET_SUBMISSIONS]:
        try:
            ws = sp.worksheet(sheet_name)
            all_vals = ws.get_all_values()
            if not all_vals:
                continue
            headers  = all_vals[0]
            team_col = headers.index("team")
            rows_to_delete = [
                i + 2
                for i, row in enumerate(all_vals[1:])
                if row[team_col] == team_name
            ]
            for row_idx in reversed(rows_to_delete):
                ws.delete_rows(row_idx)
        except Exception:
            pass


def clear_all_entries():
    sp = get_spreadsheet()
    if sp is None:
        return
    for sheet_name, headers in [
        (SHEET_LEADERBOARD, LEADERBOARD_HEADERS),
        (SHEET_SUBMISSIONS, SUBMISSION_HEADERS),
    ]:
        try:
            ws = sp.worksheet(sheet_name)
            ws.clear()
            ws.append_row(headers)
        except Exception:
            pass

# =============================================================================
#   SECRET LABELS
# =============================================================================

def load_secret_labels():
    try:
        secret_csv = st.secrets["secret_labels_csv"]
        df = pd.read_csv(StringIO(secret_csv))
        cols = [c for c in ["clip_id", "language", "is_native", "accent_region"] if c in df.columns]
        if len(df) > 0:
            return df[cols]
    except Exception:
        pass

    from pathlib import Path
    base = Path(__file__).parent
    for path in [base / "organizer" / "secret_test_labels.csv",
                 base / "secret_test_labels.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            cols = [c for c in ["clip_id", "language", "is_native", "accent_region"] if c in df.columns]
            return df[cols]
    return None

# =============================================================================
#   VALIDATION & SCORING
# =============================================================================

def validate_submission(df: pd.DataFrame) -> list:
    errors = []
    required = ["clip_id", "language", "is_native",
                "accent_region", "confidence_language", "confidence_native"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
        return errors
    invalid_langs = set(df["language"].unique()) - set(LANGUAGES)
    if invalid_langs:
        errors.append(f"Invalid language codes: {invalid_langs}")
    if not df["is_native"].isin([0, 1]).all():
        errors.append("Column 'is_native' must contain only 0 or 1.")
    for col in ["confidence_language", "confidence_native"]:
        if col in df.columns and not ((df[col] >= 0) & (df[col] <= 1)).all():
            errors.append(f"Column '{col}' must be in [0.0, 1.0].")
    if df["clip_id"].duplicated().any():
        errors.append("Duplicate clip_id values found.")
    return errors


def compute_score(submission: pd.DataFrame, secret: pd.DataFrame) -> dict:
    sub    = submission[["clip_id", "language", "is_native"]].copy()
    sec    = secret[["clip_id", "language", "is_native"]].copy()
    merged = sub.merge(sec, on="clip_id", suffixes=("_pred", "_true"))

    if len(merged) == 0:
        return {"error": "No matching clip_id found."}

    coverage = len(merged) / len(secret) * 100
    f1  = f1_score(merged["language_true"], merged["language_pred"],
                   average="weighted", zero_division=0)
    acc = accuracy_score(merged["is_native_true"], merged["is_native_pred"])

    per_lang = {}
    for lang in LANGUAGES:
        mask = merged["language_true"] == lang
        if mask.sum() > 0:
            per_lang[lang] = round(float(f1_score(
                merged.loc[mask, "language_true"],
                merged.loc[mask, "language_pred"],
                average="weighted", zero_division=0, labels=[lang]
            )), 4)

    return {
        "f1_language":    round(float(f1),  4),
        "acc_native":     round(float(acc), 4),
        "final_score":    round(0.6 * f1 + 0.4 * acc, 4),
        "coverage":       round(coverage, 1),
        "n_predictions":  len(merged),
        "per_language_f1": per_lang,
        "error": None,
    }

# =============================================================================
#   HELPERS
# =============================================================================

def count_today_submissions(leaderboard: list, team_name: str) -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    return sum(1 for e in leaderboard
               if e["team"] == team_name and e["timestamp"].startswith(today))


def get_best_per_team(leaderboard: list) -> pd.DataFrame:
    if not leaderboard:
        return pd.DataFrame()
    df = pd.DataFrame(leaderboard)
    best = (df.sort_values("final_score", ascending=False)
              .groupby("team", as_index=False).first()
              .sort_values("final_score", ascending=False)
              .reset_index(drop=True))
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
.score-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
    border: 1px solid #2d3748; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin: 0.5rem 0;
}
.big-score {
    font-family: 'Space Mono', monospace; font-size: 2rem;
    font-weight: 700; color: #48bb78; line-height: 1;
}
.score-label {
    font-size: 0.8rem; color: #718096;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;
}
.top-team { background: linear-gradient(135deg, #1a2f1a 0%, #0f1117 100%); border: 1px solid #276749; }
.stButton > button {
    background: linear-gradient(135deg, #276749, #48bb78); color: white;
    border: none; border-radius: 8px; font-family: 'Space Mono', monospace;
    font-weight: 700; padding: 0.6rem 2rem; width: 100%; font-size: 1rem;
}
.warning-box { background:#2d2000;border:1px solid #d69e2e;border-radius:8px;padding:.8rem 1rem;color:#fbd38d;font-size:.9rem; }
.success-box { background:#1a2f1a;border:1px solid #48bb78;border-radius:8px;padding:.8rem 1rem;color:#9ae6b4;font-size:.9rem; }
.error-box   { background:#2d1515;border:1px solid #fc8181;border-radius:8px;padding:.8rem 1rem;color:#feb2b2;font-size:.9rem; }
.info-banner { background:#1a2535;border:1px solid #4299e1;border-radius:8px;padding:.8rem 1rem;color:#90cdf4;font-size:.9rem;margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"# {COMPETITION_TITLE}")
st.markdown(f"*{COMPETITION_SUBTITLE}*")
st.divider()

with st.sidebar:
    st.markdown("## 📋 Competition Info")
    st.markdown(f"""
| | |
|---|---|
| **Languages** | {len(LANGUAGES)} |
| **Train clips** | 3 360 |
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
    st.divider()
    st.markdown("### 💾 Backend")
    if not GSPREAD_OK:
        st.error("gspread not installed")
    else:
        sp_check = get_spreadsheet()
        if sp_check:
            st.success("✅ Google Sheets connected")
        else:
            st.error("❌ Not connected")

tab_lb, tab_submit, tab_analysis, tab_admin = st.tabs([
    "🏆 Leaderboard", "📤 Submit", "📊 Analysis", "⚙️ Admin"
])

leaderboard = load_leaderboard()

# ══════════════════════════════════════════════════════════════════════════════
with tab_lb:
    st.markdown("## 🏆 Leaderboard — Best Score per Team")
    best_df = get_best_per_team(leaderboard)

    if best_df.empty:
        st.info("No submissions yet. Be the first to submit!")
    else:
        top3   = best_df.head(3)
        medals = ["🥇", "🥈", "🥉"]
        cols   = st.columns(min(3, len(top3)))

        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class="score-card {'top-team' if i == 0 else ''}">
                    <div class="score-label">{medals[i]} Rank {i+1}</div>
                    <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:.5rem">{row['team']}</div>
                    <div class="big-score">{row['final_score']:.2%}</div>
                    <div style="margin-top:.8rem;font-size:.85rem;color:#718096">
                        F1 Lang: <b style="color:#90cdf4">{row['f1_language']:.2%}</b> &nbsp;|&nbsp;
                        Acc Native: <b style="color:#90cdf4">{row['acc_native']:.2%}</b>
                    </div>
                    <div style="font-size:.75rem;color:#4a5568;margin-top:.4rem">{row['timestamp']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("### 📋 Full Rankings")
        display_df = best_df[["team", "final_score", "f1_language",
                               "acc_native", "coverage", "timestamp"]].copy()
        display_df.columns = ["Team", "Final Score", "F1 Language",
                               "Acc Native", "Coverage %", "Best Submission"]
        display_df.insert(0, "Rank",
            [medals[i] if i < 3 else f"#{i+1}" for i in range(len(display_df))])
        display_df = display_df.reset_index(drop=True)

        # fmt = {"Final Score": "{:.4f}", "F1 Language": "{:.4f}",
        #       "Acc Native": "{:.4f}", "Coverage %": "{:.1f}%"}
      
        fmt = {"Final Score": "{:.2%}", "F1 Language": "{:.2%}",
               "Acc Native": "{:.2%}", "Coverage %": "{:.1f}%"}
      
        score_min = display_df["Final Score"].min()
        score_max = display_df["Final Score"].max()
        if score_min == score_max:
            styled = display_df.style.format(fmt)
        else:
            styled = (display_df.style
                      .background_gradient(subset=["Final Score"], cmap="Greens",
                                           vmin=score_min, vmax=score_max)
                      .format(fmt))
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if st.button("🔄 Refresh Leaderboard"):
            st.cache_resource.clear()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
with tab_submit:
    st.markdown("## 📤 Submit Your Predictions")

    if not SKLEARN_OK:
        st.error("scikit-learn not installed.")
        st.stop()
    if not GSPREAD_OK or get_spreadsheet() is None:
        st.error("❌ Google Sheets not connected. Check your secrets configuration.")
        st.stop()

    secret_df = load_secret_labels()
    if secret_df is None:
        st.error("❌ Secret labels not configured.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input("👤 Team / Participant name", placeholder="e.g. Team AlphaWave")
    with col2:
        uploaded = st.file_uploader("📁 Upload submission.csv", type=["csv"])

    if team_name and uploaded:
        n_today   = count_today_submissions(leaderboard, team_name)
        remaining = MAX_SUBMISSIONS_DAY - n_today

        if remaining <= 0:
            st.markdown(f"""<div class="warning-box">
                ⚠️ <b>{team_name}</b> has reached the daily limit ({MAX_SUBMISSIONS_DAY}/day).
                Come back tomorrow!</div>""", unsafe_allow_html=True)
        else:
            st.info(f"📊 Submissions today: **{n_today}/{MAX_SUBMISSIONS_DAY}** — {remaining} remaining")

            if st.button("🚀 Score My Submission"):
                with st.spinner("Scoring and saving to Google Sheets…"):
                    try:
                        sub_df = pd.read_csv(uploaded)
                        errors = validate_submission(sub_df)

                        if errors:
                            st.markdown(f"""<div class="error-box">
                                <b>❌ Submission format errors:</b><br>
                                {"<br>".join(f"• {e}" for e in errors)}
                            </div>""", unsafe_allow_html=True)
                        else:
                            scores = compute_score(sub_df, secret_df)
                            if scores.get("error"):
                                st.error(scores["error"])
                            else:
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                save_submission_csv(team_name, ts, uploaded.name, sub_df)
                                entry = {
                                    "team": team_name, "timestamp": ts,
                                    "final_score": scores["final_score"],
                                    "f1_language": scores["f1_language"],
                                    "acc_native":  scores["acc_native"],
                                    "coverage":    scores["coverage"],
                                    "n_predictions": scores["n_predictions"],
                                    "per_language_f1": scores.get("per_language_f1", {}),
                                    "filename": uploaded.name,
                                }
                                append_leaderboard_entry(entry)
                                leaderboard.append(entry)

                                st.markdown('<div class="success-box">✅ <b>Scored and saved to Google Sheets!</b></div>',
                                            unsafe_allow_html=True)
                                st.markdown("### 🎯 Your Scores")
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.markdown(f"""<div class="score-card">
                                        <div class="score-label">🏆 Final Score</div>
                                        <div class="big-score">{scores['final_score']:.4f}</div>
                                        <div style="color:#718096;font-size:.8rem">0.6×F1 + 0.4×Acc</div>
                                    </div>""", unsafe_allow_html=True)
                                with c2:
                                    st.markdown(f"""<div class="score-card">
                                        <div class="score-label">🌍 F1 Language</div>
                                        <div class="big-score" style="color:#90cdf4">{scores['f1_language']:.4f}</div>
                                        <div style="color:#718096;font-size:.8rem">weighted F1-score</div>
                                    </div>""", unsafe_allow_html=True)
                                with c3:
                                    st.markdown(f"""<div class="score-card">
                                        <div class="score-label">🎙 Acc Native</div>
                                        <div class="big-score" style="color:#fbd38d">{scores['acc_native']:.4f}</div>
                                        <div style="color:#718096;font-size:.8rem">binary accuracy</div>
                                    </div>""", unsafe_allow_html=True)

                                best_new  = get_best_per_team(leaderboard)
                                rank_rows = best_new[best_new["team"] == team_name]
                                if len(rank_rows) > 0:
                                    r = rank_rows.index[0]
                                    medal = ["🥇","🥈","🥉"][r-1] if r <= 3 else f"#{r}"
                                    st.success(f"Your current rank: **{medal} Position {r}** out of {len(best_new)} teams")

                                if scores.get("per_language_f1"):
                                    st.markdown("### 🌍 Per-Language F1 Breakdown")
                                    lang_df = pd.DataFrame(
                                        [(LANG_NAMES.get(k, k), v)
                                         for k, v in scores["per_language_f1"].items()],
                                        columns=["Language", "F1 Score"]
                                    ).sort_values("F1 Score", ascending=False)
                                    st.bar_chart(lang_df.set_index("Language"))
                    except Exception as e:
                        st.error(f"Error: {e}")

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
            st.markdown("### 📈 Best Score per Team")
            st.bar_chart(df_all.groupby("team")["final_score"].max().sort_values(ascending=False))
        with col2:
            st.markdown("### 📅 Submissions Over Time")
            st.bar_chart(df_all.groupby(df_all["timestamp"].dt.date).size())
        st.markdown("### 📋 All Submissions History")
        history = (df_all[["team","timestamp","final_score","f1_language","acc_native","filename"]]
                   .copy().sort_values("timestamp", ascending=False))
        history["timestamp"] = history["timestamp"].astype(str)
        st.dataframe(history, use_container_width=True)
        st.markdown("### 📊 Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Submissions", len(df_all))
        c2.metric("Teams",             df_all["team"].nunique())
        c3.metric("Best Score",        f"{df_all['final_score'].max():.4f}")
        c4.metric("Average Score",     f"{df_all['final_score'].mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    st.markdown("## ⚙️ Admin Panel")
    if not IS_ADMIN:
        st.warning("Enter the admin password in the sidebar.")
    else:
        st.success("✓ Admin access granted")

        st.markdown("### 📊 Google Sheets")
        try:
            url = st.secrets["spreadsheet_url"]
            st.markdown(f"🔗 [Open Google Sheet]({url})")
        except Exception:
            pass
        st.info(f"Total entries: **{len(leaderboard)}**")

        st.divider()
        st.markdown("### 🚫 Remove a Team")
        if leaderboard:
            teams = sorted(set(e["team"] for e in leaderboard))
            team_to_remove = st.selectbox("Select team", teams)
            if st.button(f"Remove '{team_to_remove}'"):
                with st.spinner(f"Removing {team_to_remove}…"):
                    delete_team_entries(team_to_remove)
                st.success(f"Team '{team_to_remove}' removed.")
                st.rerun()

        st.divider()
        st.markdown("### 🗑️ Reset Everything")
        if st.button("⚠️ Clear ALL submissions", type="primary"):
            with st.spinner("Clearing…"):
                clear_all_entries()
            st.success("All submissions cleared.")
            st.rerun()

        st.divider()
        st.markdown("### 📥 Export")
        if leaderboard:
            best = get_best_per_team(leaderboard)
            st.download_button("⬇️ Download leaderboard.csv",
                               data=best.to_csv(index=True),
                               file_name="final_leaderboard.csv",
                               mime="text/csv")

        st.divider()
        st.markdown("### 🔑 Admin Password")
        st.info("Set in `.streamlit/secrets.toml`:\n```toml\nadmin_password = 'your_password'\n```")
