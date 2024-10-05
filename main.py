import csv
import os
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
# .envファイルからAPIキーをロード
load_dotenv()
# openai.api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)
# CSVファイルからサークル情報を読み込む関数
def load_circle_info(file_path):
    circle_data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            circle_data.append(row)
    return circle_data

# OpenAIのAPIを使ってテキストをベクトル化する関数
def get_embedding(text):
    response =  client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"  # 埋め込みモデルの指定
    )
    return response.data[0].embedding

def determine_k(question, fields):
    embeddings = [get_embedding(field) for field in fields]
    
    # 質問をベクトル化
    question_vector = np.array([get_embedding(question)], dtype='float32')

    # フィールド名のベクトルをFaiss用に変換
    field_vectors = np.array(embeddings, dtype='float32')

    # FaissでL2距離のインデックス作成
    dimension = 1536  # text-embedding-ada-002のベクトル次元
    index = faiss.IndexFlatL2(dimension)
    index.add(field_vectors)

    # 最も近いフィールドを検索（上位n個のフィールドを探す）
    _, indices = index.search(question_vector, k=len(fields))  # 全てのフィールドに対して距離を計算
    
    # 意味的に最も近いフィールドを動的にフィルタリング
    matched_fields = [fields[i] for i in indices[0] if fields[i] in question]

    # マッチしたフィールド数（k）を返す
    return len(matched_fields) if len(matched_fields) > 0 else 1 

# Faissを使って質問に最も関連性の高いフィールドを見つける関数
def find_closest_fields(question, fields, k):
    # 質問のベクトル化
    question_vector = np.array([get_embedding(question)], dtype='float32')

    # フィールド名をベクトル化
    field_vectors = np.array([get_embedding(field) for field in fields], dtype='float32')

    # FaissでL2距離のインデックス作成
    dimension = 1536  # text-embedding-ada-002のベクトル次元
    index = faiss.IndexFlatL2(dimension)
    index.add(field_vectors)
    # print(field_vectors)
    # 質問に最も関連するフィールドを複数検索
    _, indices = index.search(question_vector, k=k)  # k個の近いフィールドを返す
    
    # 見つかったフィールドを返す
    return [fields[idx] for idx in indices[0]]


# 必要な情報をCSVデータから取得する関数
def get_circle_info(circle_data, required_fields):
    results = []
    for circle in circle_data:
        info = {field: circle[field] for field in required_fields if field in circle}
        if info:
            results.append(info)
    return results


# GPT APIを使って自然な応答を生成
def generate_response(circle_info, question):
    print(circle_info)
    prompt = f"ユーザーの質問: '{question}' に基づいて、以下のサークル情報を参考にして人間らしい返答を作成してください: '{circle_info}'"
    response = client.chat.completions.create(
        model="gpt-4",  # 使用するGPTモデルを指定
        messages=[
            {"role": "system", "content": "あなたはサークルの情報について回答するAIです。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100  # 必要に応じてトークン数を調整
    )
    return response.choices[0].message.content

# メイン処理
def main():
    # サークル情報をCSVファイルから読み込む
    circle_data = load_circle_info('test.csv')

    # ユーザーからの質問
    question = "今の代表と、男女比は？"

    if circle_data:
        available_fields = list(circle_data[0].keys())

        # 質問に最も関連するフィールド数を指定（質問内のフィールド数に応じてkを指定）
        # k = len(available_fields)  # フィールドの数に基づいて動的に取得する
        # k = 10
        k = determine_k(question, available_fields)
        k = 2
        print(k)
        # 質問に最も近いフィールドを複数ベクトル検索で特定
        closest_fields = find_closest_fields(question, available_fields, k=k)
        print(closest_fields)
        if closest_fields:
            # 必要な情報をCSVから取得
            circle_info = get_circle_info(circle_data, closest_fields)
            
            # ChatGPTを使って人間らしい文章を生成
            response = generate_response(circle_info, question)
            
            # 結果を表示
            print(f"質問: {question}")
            print(f"生成された応答: {response}")
        else:
            print("質問に対応する情報が見つかりませんでした。")
    else:
        print("サークル情報がありません。")

# 実行
if __name__ == "__main__":
    main()
