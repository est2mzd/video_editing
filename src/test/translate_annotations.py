from openai import OpenAI
import json

client = OpenAI()

input_path = '/workspace/data/annotations.jsonl'
output_path = '/workspace/data/instructions_ja.csv'

with open(input_path) as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    data = json.loads(line)
    text = data["instruction"]
    video_path = data["video_path"]
    res = client.chat.completions.create(
        #model="gpt-5.3",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "英語を自然な日本語に逐次翻訳せよ。省略禁止。"},
            {"role": "user", "content": text}
        ]
    )

    # ヘッダを記入
    if i == 0:
        with open(output_path, "w") as out_f:
            out_f.write("video_path,instruction\n")

    with open(output_path, "a") as out_f:
        out_f.write(f"{video_path},{res.choices[0].message.content}\n")
