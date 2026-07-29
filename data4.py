import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------
# 1. 設定
# ----------------------------------------
MODEL_NAME = 'llama3.1:latest'
TOTAL_PEOPLE = 10      # 1試行あたりの伝搬人数

# CSVファイルを読み込み
df = pd.read_csv('dataset.csv', encoding='utf-8')

# 【重要】データ数（50件）を自動取得し、全件で処理を行う
NUM_TRIALS = len(df)
print(f"=== 全件バッチ処理実験開始（総データ数: {NUM_TRIALS}件） ===")

all_lengths = []
all_similarities = []

# ----------------------------------------
# 2. 50件の全件ループ（重複なし）
# ----------------------------------------
for idx, row in df.iterrows():
    trial_num = idx + 1
    print(f"\n--- 処理中 {trial_num}/{NUM_TRIALS} (ID: {row['id']}) ---")
    
    text_origin = row['text']
    trial_texts = [text_origin]
    trial_lengths = [len(text_origin)]
    
    # 10人連続伝搬
    current_text = text_origin
    for i in range(1, TOTAL_PEOPLE + 1):
        prompt = (
            "あなたは伝言ゲームのプレイヤーです。前の人が言った文章の重要な意味や含まれる事実を維持しつつ、"
            "文章量は半分に（冗長な表現を徹底的に排除）していってください。\n"
            "※余計な挨拶や解説は絶対に含めず、要約した文章だけを出力してください。\n\n"
            f"前の人の文章：{current_text}"
        )
        
        try:
            response = ollama.generate(model=MODEL_NAME, prompt=prompt)
            output_text = response['response'].strip().replace("\n", "")
            if not output_text:
                output_text = "省略"
        except Exception as e:
            print(f"Ollama Error: {e}")
            output_text = current_text
            
        trial_texts.append(output_text)
        trial_lengths.append(len(output_text))
        current_text = output_text
    
    # 類似度計算 (TF-IDF N-gram)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    vectors = vectorizer.fit_transform(trial_texts).toarray()
    
    origin_vector = vectors[0].reshape(1, -1)
    trial_sims = [1.0]
    for i in range(1, len(vectors)):
        current_vector = vectors[i].reshape(1, -1)
        sim = cosine_similarity(origin_vector, current_vector)[0][0]
        trial_sims.append(sim)
        
    all_lengths.append(trial_lengths)
    all_similarities.append(trial_sims)

# ----------------------------------------
# 3. 統計処理とCSV保存
# ----------------------------------------
labels = [f"{i}人目" if i > 0 else "原文" for i in range(TOTAL_PEOPLE + 1)]
mean_lengths = np.mean(all_lengths, axis=0)
mean_sims = np.mean(all_similarities, axis=0)

df_summary = pd.DataFrame({
    "プレイヤー": labels,
    "平均文字数": mean_lengths,
    "平均対原文類似度": mean_sims
})
df_summary.to_csv("all_50_samples_summary.csv", index=False, encoding='utf-8-sig')
print("\n[情報] 50件の平均データを 'all_50_samples_summary.csv' に保存しました。")

# ----------------------------------------
# 4. グラフの描画
# ----------------------------------------
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    plt.rcParams['font.family'] = 'sans-serif'

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('伝搬した人数')
ax1.set_ylabel('平均コサイン類似度（文脈の比較）', color=color)
ax1.plot(labels, mean_sims, marker='o', color=color, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.0, 1.05)

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('平均文字数（テキストの長さ）', color=color)
ax2.plot(labels, mean_lengths, marker='s', linestyle='--', color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f'LLMによる10人伝言ゲーム：全{NUM_TRIALS}件の平均情報減衰曲線')
fig.tight_layout()
plt.savefig('propagation_all_50_result.png')
print("[情報] グラフを 'propagation_all_50_result.png' として保存しました。")
plt.show()
