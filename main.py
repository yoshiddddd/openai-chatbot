import openai
import faiss
import numpy as np
import os
from dotenv import load_dotenv
# OpenAI APIキーの設定
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# サークル情報のファイルを読み込む
def load_circle_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"  # 埋め込みモデルの指定
    )
    return response.data[0].embedding
    # return response['data'][0]['embedding']

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

    # サークル情報をベクトル化してFaissに保存
    index = store_in_faiss(circle_info)

    # ユーザーからの質問
    question = "サークルのメンバーは何人ですか？"

    # 質問に対して最も関連性の高いサークル情報を検索
    result, distance = search_faiss(index, question, circle_info)
    
    # 検索結果を表示
    print(f"質問: {question}")
    print(f"関連するサークル情報: {result} (距離: {distance})")

# 実行
if __name__ == "__main__":
    main()
