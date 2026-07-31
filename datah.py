import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ollama

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import beta


# ============================================================
# 1. 設定
# ============================================================

MODEL_NAME = "llama3.1:latest"

TOTAL_PEOPLE = 10

# 50件使用
NUM_TRIALS = 50

# 同時に実行する試行数
# GPU負荷が高い場合は 2～4 を推奨
# CPU中心なら 4～8 も可能
MAX_WORKERS = 4


# ============================================================
# 2. データ読み込み
# ============================================================

df = pd.read_csv("dataset.csv", encoding="utf-8")

# 50件使用
df_prototype = df.head(NUM_TRIALS)

NUM_TRIALS = len(df_prototype)

print(
    f"=== 4条件LLM並列実験開始 ===\n"
    f"データ数: {NUM_TRIALS}件\n"
    f"伝搬人数: {TOTAL_PEOPLE}人\n"
    f"並列数: {MAX_WORKERS}\n"
)


# ============================================================
# 3. 条件設定
# ============================================================

CONDITIONS = {

    "そのまま": {
        "prompt_template": """あなたは伝言ゲームのプレイヤーです。
前の人が伝えた文章の「事実関係」や「核心的な意味」をすべて維持したまま、次の人に同じ内容を伝達してください。

以下のルールを必ず守ってください。
- 出力する文章の全体の長さ（文字数）は、入力された文章とほぼ同じにしてください。
- 前の人の文章に含まれる事実、人物、出来事、因果関係を削除しないでください。
- 新しい事実を追加しないでください。
- 一言一句同じにする必要はありません。
- 同じ情報量・同じ長さを保った自然な日本語にしてください。
- 挨拶、説明、前置き、箇条書き、引用符を追加しない。

文章だけを出力してください。

【前の人の文章】
{current_text}"""
    },


    "要約": {
        "prompt_template": """あなたは伝言ゲームのプレイヤーです。
前の人が伝えた文章を、元の文章の重要な事実関係と意味をできるだけ維持したまま要約してください。

以下のルールを必ず守ってください。
- 重要な事実、人物、出来事、因果関係を削除しない
- 新しい事実や推測を追加しない
- 内容を言い換えてよい
- 冗長な表現を削除する
- 入力文章のおよそ50%の文字数を目標とする
- 意味の維持を優先する
- 挨拶、説明、前置き、箇条書き、引用符を追加しない

文章だけを出力してください。

【前の人の文章】
{current_text}"""
    },


    "助長": {
        "prompt_template": """あなたは伝言ゲームのプレイヤーです。
前の人が伝えた文章の事実関係と核心的な意味を維持したまま、文章をより詳しく表現してください。

以下のルールを必ず守ってください。
- 元の文章に含まれる事実関係を変更しない
- 新しい事実、人物、数字、出来事、推測を追加しない
- 説明的な表現や修飾表現を増やす
- 同じ内容をより詳細な表現に言い換える
- 入力文章のおよそ150%の文字数を目標とする
- 事実関係の維持を優先する
- 挨拶、説明、前置き、箇条書き、引用符を追加しない

文章だけを出力してください。

【前の人の文章】
{current_text}"""
    },


    "自己修復": {
        "prompt_template": """あなたは伝言ゲームのプレイヤーです。

【原文】を正しい情報源として使用し、
【前の人の文章】に含まれる情報の誤りや欠落を可能な範囲で修正してください。

そのうえで、原文の重要な事実関係と意味を維持した短い文章に要約してください。

以下のルールを必ず守ってください。
- 【原文】に記載されている事実を正しい情報として扱う
- 【前の人の文章】に誤りがある場合は修正する
- 欠落している重要情報があれば原文から補完する
- 原文に存在しない情報を作らない
- 原文の重要な意味と事実関係を維持する
- 入力文章のおよそ50%の文字数を目標とする
- 情報の正確性を優先する
- 挨拶、説明、前置き、箇条書き、引用符を追加しない

修復・要約した文章だけを出力してください。

【原文】
{text_origin}

【前の人の文章】
{current_text}"""
    }
}


# ============================================================
# 4. 1試行分を処理する関数
# ============================================================

def run_trial(cond_name, cond_info, idx, row):

    trial_num = idx + 1

    current_id = row["id"]
    text_origin = str(row["text"])

    trial_texts = [text_origin]
    trial_lengths = [len(text_origin)]

    current_text = text_origin

    # --------------------------------------------------------
    # 10人の伝搬
    # ※ここは逐次処理
    # --------------------------------------------------------

    for person in range(1, TOTAL_PEOPLE + 1):

        if cond_name == "自己修復":

            prompt = cond_info["prompt_template"].format(
                text_origin=text_origin,
                current_text=current_text
            )

        else:

            prompt = cond_info["prompt_template"].format(
                current_text=current_text
            )

        try:

            response = ollama.generate(
                model=MODEL_NAME,
                prompt=prompt
            )

            output_text = response["response"].strip()
            output_text = output_text.replace("\n", "")

            if not output_text:
                output_text = "省略"

        except Exception as e:

            print(
                f"[ERROR] 条件={cond_name}, "
                f"ID={current_id}, "
                f"{person}人目: {e}"
            )

            output_text = current_text

        trial_texts.append(output_text)
        trial_lengths.append(len(output_text))

        current_text = output_text

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 3)
    )

    vectors = vectorizer.fit_transform(trial_texts)

    origin_vector = vectors[0]

    trial_sims = [1.0]

    for i in range(1, len(trial_texts)):

        sim = cosine_similarity(
            origin_vector,
            vectors[i]
        )[0][0]

        trial_sims.append(sim)

    # --------------------------------------------------------
    # 結果をまとめる
    # --------------------------------------------------------

    detailed = []

    detailed.append({
        "条件": cond_name,
        "サンプルのインデックス": trial_num,
        "ID": current_id,
        "プレイヤー": "原文",
        "文字数": len(text_origin),
        "対原文類似度": 1.0,
        "テキスト内容": text_origin
    })

    for i in range(1, len(trial_texts)):

        detailed.append({
            "条件": cond_name,
            "サンプルのインデックス": trial_num,
            "ID": current_id,
            "プレイヤー": f"{i}人目",
            "文字数": trial_lengths[i],
            "対原文類似度": trial_sims[i],
            "テキスト内容": trial_texts[i]
        })

    return {
        "idx": idx,
        "id": current_id,
        "lengths": trial_lengths,
        "sims": trial_sims,
        "detailed": detailed
    }


# ============================================================
# 5. 並列実験
# ============================================================

summary_results = {}

detailed_records = []


for cond_name, cond_info in CONDITIONS.items():

    print()
    print("=" * 60)
    print(f"実験条件: 【{cond_name}】")
    print("=" * 60)

    cond_results = []

    # --------------------------------------------------------
    # 並列実行
    # --------------------------------------------------------

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = []

        for idx, row in df_prototype.iterrows():

            future = executor.submit(
                run_trial,
                cond_name,
                cond_info,
                idx,
                row
            )

            futures.append(future)

        # ----------------------------------------------------
        # 結果回収
        # ----------------------------------------------------

        for count, future in enumerate(
            as_completed(futures),
            start=1
        ):

            result = future.result()

            cond_results.append(result)

            detailed_records.extend(
                result["detailed"]
            )

            print(
                f"[{cond_name}] "
                f"{count}/{NUM_TRIALS} 完了"
            )

    # --------------------------------------------------------
    # 元の順番に戻す
    # --------------------------------------------------------

    cond_results.sort(
        key=lambda x: x["idx"]
    )

    cond_lengths = [
        r["lengths"]
        for r in cond_results
    ]

    cond_similarities = [
        r["sims"]
        for r in cond_results
    ]

    summary_results[cond_name] = {

        "mean_lengths":
            np.mean(cond_lengths, axis=0),

        "mean_sims":
            np.mean(cond_similarities, axis=0),

        "raw_sims":
            np.array(cond_similarities)
    }


# ============================================================
# 6. 詳細CSV
# ============================================================

df_detailed_output = pd.DataFrame(
    detailed_records
)

df_detailed_output.to_csv(
    "llm_experiment_full_details.csv",
    index=False,
    encoding="utf-8-sig"
)

print()
print(
    "[情報] "
    "llm_experiment_full_details.csv を保存しました。"
)


# ============================================================
# 7. ベータ分布統計
# ============================================================

summary_rows = []

labels = [
    f"{i}人目" if i > 0 else "原文"
    for i in range(TOTAL_PEOPLE + 1)
]


print(
    "[情報] ベータ分布への最尤フィッティングを計算中..."
)


for cond_name, data in summary_results.items():

    for p_idx, label in enumerate(labels):

        mean_len = data["mean_lengths"][p_idx]

        mean_sim = data["mean_sims"][p_idx]

        error_rates = (
            1.0 - data["raw_sims"][:, p_idx]
        )

        error_rates = np.clip(
            error_rates,
            1e-5,
            1 - 1e-5
        )

        if len(error_rates) < 3:

            alpha_hat = np.nan
            beta_hat = np.nan

        elif np.var(error_rates) == 0:

            alpha_hat = np.nan
            beta_hat = np.nan

        else:

            try:

                alpha_hat, beta_hat, _, _ = beta.fit(
                    error_rates,
                    floc=0,
                    fscale=1
                )

            except Exception:

                alpha_hat = np.nan
                beta_hat = np.nan

        summary_rows.append({

            "条件": cond_name,

            "プレイヤー": label,

            "平均文字数": mean_len,

            "平均対原文類似度": mean_sim,

            "ベータ分布_alpha": alpha_hat,

            "ベータ分布_beta": beta_hat

        })


df_summary_output = pd.DataFrame(
    summary_rows
)

df_summary_output.to_csv(
    "llm_conditions_summary_statistics.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "[情報] "
    "llm_conditions_summary_statistics.csv を保存しました。"
)


# ============================================================
# 8. ケーススタディ
# ============================================================

summary_cond = summary_results["要約"]

sum3_scores = summary_cond["raw_sims"][:, 3]

best_idx = np.argmax(sum3_scores)

worst_idx = np.argmin(sum3_scores)

best_id = df_prototype.iloc[best_idx]["id"]

worst_id = df_prototype.iloc[worst_idx]["id"]


with open(
    "case_study_output.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 50 + "\n")
    f.write("ケーススタディ\n")
    f.write("=" * 50 + "\n\n")

    for label_name, target_id in [

        ("パターンA（高スコア）", best_id),

        ("パターンB（低スコア）", worst_id)

    ]:

        f.write(
            f"【{label_name} ID:{target_id}】\n"
        )

        sub_df = df_detailed_output[
            (df_detailed_output["条件"] == "要約") &
            (df_detailed_output["ID"] == target_id)
        ]

        for _, r in sub_df.iterrows():

            f.write(
                f"{r['プレイヤー']} "
                f"({r['文字数']}文字 / "
                f"類似度:{r['対原文類似度']:.4f}): "
                f"{r['テキスト内容']}\n"
            )

        f.write("\n" + "-" * 50 + "\n\n")


# ============================================================
# 9. グラフ
# ============================================================

try:
    plt.rcParams["font.family"] = "MS Gothic"
except:
    plt.rcParams["font.family"] = "sans-serif"


fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(16, 6)
)


markers = {
    "そのまま": "o",
    "要約": "s",
    "助長": "^",
    "自己修復": "d"
}


colors = {
    "そのまま": "tab:green",
    "要約": "tab:blue",
    "助長": "tab:red",
    "自己修復": "tab:purple"
}


for cond_name, data in summary_results.items():

    ax1.plot(
        labels,
        data["mean_sims"],
        marker=markers[cond_name],
        color=colors[cond_name],
        linewidth=2,
        label=cond_name
    )


ax1.set_ylim(0.0, 1.05)

ax1.set_title(
    f"各条件における平均コサイン類似度"
    f"（{NUM_TRIALS}件平均）"
)

ax1.legend()

ax1.grid(
    True,
    linestyle="--",
    alpha=0.6
)


for cond_name, data in summary_results.items():

    ax2.plot(
        labels,
        data["mean_lengths"],
        marker=markers[cond_name],
        color=colors[cond_name],
        linestyle="--",
        linewidth=2,
        label=cond_name
    )


ax2.set_title(
    f"各条件における平均文字数の推移"
    f"（{NUM_TRIALS}件平均）"
)

ax2.legend()

ax2.grid(
    True,
    linestyle="--",
    alpha=0.6
)


plt.suptitle(
    f"LLM伝言ゲーム実験結果"
    f"（{NUM_TRIALS}件）",
    fontsize=14
)

fig.tight_layout()

plt.savefig(
    "propagation_4_conditions_comparison.png"
)

plt.show()

print()
print("=== 実験完了 ===")