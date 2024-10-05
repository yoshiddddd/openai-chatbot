# import openai
from openai import OpenAI
import faiss
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
# openai.api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# サークル情報のファイルを読み込む
def load_circle_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# GPT APIを使ってテキストをベクトル化
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"  # 埋め込みモデルの指定
    )
    return response.data[0].embedding

# GPT APIを使って自然な応答を生成
def generate_response(circle_info, question):
    prompt = f"ユーザーの質問: '{question}' に基づいて、以下のサークル情報を参考にして人間らしい返答を作成してください: '{circle_info}'"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用するGPTモデルを指定
        messages=[
            {"role": "system", "content": "あなたはサークルの情報について回答するAIです。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150  # 必要に応じてトークン数を調整
    )
    # return response.choices[0].message['content'].strip()
    return response.choices[0].message.content

# サークル情報をベクトル化してFaissに保存
def store_in_faiss(circle_info):
    # ベクトルの次元数
    dimension = 1536  
    # L2距離でインデックスを作成
    index = faiss.IndexFlatL2(dimension)

    # サークル情報をベクトル化し、Faissに追加
    embeddings = []
    for info in circle_info:
        embedding = get_embedding(info)
        embeddings.append(embedding)
    
    # ベクトルをNumPy配列に変換
    embeddings_np = np.array(embeddings, dtype='float32')

    # Faissにベクトルを追加
    index.add(embeddings_np)
    
    return index

# Faissで質問に対する最も近いサークル情報を検索
def search_faiss(index, question, circle_info):
    # 質問をベクトル化
    query_vector = np.array([get_embedding(question)], dtype='float32')
    
    # 類似ベクトルを検索
    distances, indices = index.search(query_vector, k=1)  # k=1は最も近いベクトルを1つ返す
    return circle_info[indices[0][0]], distances[0][0]

# メイン処理
def main():
    # サークル情報をテキストファイルから読み込む
    circle_info = load_circle_info('circles.txt')
    print(f"サークル情報: {circle_info}")
    
    # サークル情報をベクトル化してFaissに保存
    index = store_in_faiss(circle_info)

    # ユーザーからの質問
    question = "サークルの創設者と代表を教えて"

    # 質問に対して最も関連性の高いサークル情報を検索
    closest_info, distance = search_faiss(index, question, circle_info)
    
    # ChatGPTを使って人間らしい文章を生成
    response = generate_response(closest_info, question)
    
    # 検索結果と生成された文章を表示
    print(f"質問: {question}")
    print(f"生成された応答: {response} (距離: {distance})")

# 実行
if __name__ == "__main__":
    main()
