#테스트용 파일
import psycopg2
import dotenv
import os
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from huggingface_hub import login

class testItem(BaseModel):
    content: str

dotenv.load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# 데이터베이스 연결 설정
connection = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS")
)
queries = ["김치는 어느나라 음식인가요?"]
contents = ["김치는 한국의 음식입니다.", "짜장면은 중국의 음식입니다.", "타코야끼는 일본의 음식입니다."]

model_name = "Qwen/Qwen3-Embedding-0.6B"
model = SentenceTransformer(model_name)

query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(contents)

cursor = connection.cursor()

# 1. content 텍스트 결합
content_str = " ".join(contents)

# 2. Vector/List 데이터를 DB에 맞게 변환 (문자열 형태 '[0.1, 0.2, ...]'로 변환)
query_emb_list = str(query_embeddings[0].tolist())
docu_emb_list = str(document_embeddings[0].tolist())

# 3. 안전한 파라미터 바인딩 (단일 인자는 (data,) 처럼 쉼표 필수)
insert_query = "INSERT INTO query_t (embed_query) VALUES (%s);"
cursor.execute(insert_query, (query_emb_list,))  # <--- 쉼표(,) 추가

insert_query2 = "INSERT INTO docu_t (content, embed) VALUES (%s, %s);"
cursor.execute(insert_query2, (content_str, docu_emb_list))

# 4. 데이터베이스 변경사항 저장
connection.commit()

print("성공적으로 DB에 저장되었습니다.")

similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)

# 5. 커서 및 연결 종료
cursor.close()
connection.close()