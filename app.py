import streamlit as st
import pandas as pd
from datetime import datetime
import os

# ==========================================================
# 基本設定
# ==========================================================
st.set_page_config(
    page_title="アンケートアプリ",
    page_icon="📝",
    layout="centered",
)

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "responses.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# アンケート項目はここで自由に編集してください
QUESTIONS = {
    "満足度": {"type": "radio", "options": ["非常に満足", "満足", "普通", "不満", "非常に不満"]},
    "利用頻度": {"type": "radio", "options": ["毎日", "週に数回", "月に数回", "ほとんど利用しない"]},
    "興味のある機能": {
        "type": "multiselect",
        "options": ["デザイン", "操作性", "価格", "サポート体制", "機能の豊富さ"],
    },
    "総合評価（5点満点）": {"type": "slider", "min": 1, "max": 5},
    "ご意見・ご要望": {"type": "text_area"},
}

COLUMNS = ["回答日時"] + list(QUESTIONS.keys())


# ==========================================================
# データ保存・読み込み
# ==========================================================
def load_responses() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=COLUMNS)


def save_response(row: dict):
    df = load_responses()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)


# ==========================================================
# 回答ページ
# ==========================================================
def render_survey_page():
    st.title("📝 アンケート")
    st.write("ご協力をお願いいたします。以下の項目にご回答ください。")

    with st.form("survey_form", clear_on_submit=True):
        answers = {}

        for label, config in QUESTIONS.items():
            qtype = config["type"]
            if qtype == "radio":
                answers[label] = st.radio(label, config["options"])
            elif qtype == "multiselect":
                answers[label] = st.multiselect(label, config["options"])
            elif qtype == "slider":
                answers[label] = st.slider(label, config["min"], config["max"])
            elif qtype == "text_area":
                answers[label] = st.text_area(label)
            elif qtype == "text_input":
                answers[label] = st.text_input(label)

        submitted = st.form_submit_button("送信する")

        if submitted:
            row = {"回答日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            for k, v in answers.items():
                row[k] = ", ".join(v) if isinstance(v, list) else v
            save_response(row)
            st.success("回答ありがとうございました！送信が完了しました。")
            st.balloons()


# ==========================================================
# 管理者ページ（結果集計）
# ==========================================================
def render_admin_page():
    st.title("📊 アンケート結果（管理者用）")

    password = st.text_input("管理者パスワードを入力してください", type="password")
    
    # st.secretsがエラーを起こさないように安全な書き方に変更
    try:
        correct_password = st.secrets.get("admin_password", "admin123")
    except Exception:
        correct_password = "admin123"  # ファイルがない場合のデフォルトパスワード

    df = load_responses()

    if df.empty:
        st.info("まだ回答が集まっていません。")
        return

    st.metric("総回答数", len(df))
    st.divider()

    for label, config in QUESTIONS.items():
        st.subheader(label)
        qtype = config["type"]
        if qtype in ("radio", "multiselect"):
            if qtype == "multiselect":
                # カンマ区切りを展開して集計
                exploded = df[label].dropna().str.split(", ").explode()
                counts = exploded.value_counts()
            else:
                counts = df[label].value_counts()
            st.bar_chart(counts)
        elif qtype == "slider":
            st.write(f"平均: {df[label].mean():.2f} / 中央値: {df[label].median():.1f}")
            st.bar_chart(df[label].value_counts().sort_index())
        elif qtype == "text_area":
            with st.expander("自由記述の回答一覧を見る"):
                for i, text in enumerate(df[label].dropna(), 1):
                    if text.strip():
                        st.write(f"{i}. {text}")

    st.divider()
    st.subheader("全回答データ")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "CSVをダウンロード",
        data=csv,
        file_name="survey_responses.csv",
        mime="text/csv",
    )


# ==========================================================
# ナビゲーション
# ==========================================================
def main():
    page = st.sidebar.radio("メニュー", ["アンケートに回答する", "結果を見る（管理者）"])

    if page == "アンケートに回答する":
        render_survey_page()
    else:
        render_admin_page()


if __name__ == "__main__":
    main()
