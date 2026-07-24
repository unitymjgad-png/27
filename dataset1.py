import pandas as pd
from ollama import chat
import time

categories = [
    "ニュース",
    "医療",
    "災害",
    "教育",
    "AI",
    "経済",
    "スポーツ",
    "環境",
    "日常",
    "歴史"
]

rows = []
id_num = 1

for category in categories:

    print(f"=== {category} ===")

    for i in range(5):

        prompt = f"""
カテゴリ：{category}


卒業研究用のデータセットを作成してください。

以下の条件を必ず守ってください。

・約100文字
・自然な日本語
・一つの出来事を説明する文章
・小説ではない
・箇条書き禁止
・タイトル禁止
・見出し禁止
・報告書禁止
・データセット説明禁止
・会話禁止
・引用符「」禁止
・文章は「。」で終わる
・説明文のみ出力する
"""

        response = chat(
            model="gemma3:4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = response["message"]["content"].replace("\n", " ").strip()

        rows.append({
            "id": id_num,
            "category": category,
            "text": text
        })

        print(id_num)

        id_num += 1

        time.sleep(0.3)

df = pd.DataFrame(rows)

df.to_csv(
    "dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("完成しました")