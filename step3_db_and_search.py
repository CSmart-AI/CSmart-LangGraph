# Cell 9
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.tools import tool
from typing import List
import pandas as pd
import os


# ======================================================
# 1⃣ Google Gemini Embeddings 초기화 (서버 환경 대응)
# ======================================================
# - Ollama 대신 Google Gemini API 사용
# - 필요: GOOGLE_API_KEY 환경 변수 설정
# - 모델: text-embedding-004 (최신 임베딩 모델)
print("Google Gemini Embeddings 모델 초기화 중...")
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # Gemini 임베딩 모델
    task_type="retrieval_document"      # 문서 검색 최적화
)


# ======================================================
# 2⃣ Chroma DB 생성 또는 불러오기
# ======================================================
persist_dir = "./chroma_guideline"
collection_name = "guideline_db"

if not os.path.exists(persist_dir):
    print(" 최초 실행: CSV에서 GuidelineDB 생성 중...")
    df = pd.read_csv("GuidelineDB.csv", encoding="utf-8")

    # CSV를 Document 리스트로 변환
    documents = []
    for _, row in df.iterrows():
        documents.append(
            Document(
                page_content=row["question"],
                metadata={
                    "answer": row["answer"],
                    "category": row.get("category", ""),
                    "source": "guidelineDB",
                    "source_name": "GuidelineDB",
                    "source_detail": row.get("출처", "출처 미기재"),
                }
            )
        )

    # 한 번에 DB 생성 및 저장
    guideline_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    print(" GuidelineDB 생성 완료 (Chroma persisted).")
else:
    print(" 기존 GuidelineDB 불러오는 중...")
    guideline_db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings_model
    )
    print(" GuidelineDB 로드 완료.")


# ======================================================
# 3⃣ Reranker 모델 설정 (생략 - LangChain 1.0 호환성 문제로 제거)
# ======================================================
# Reranker 기능은 검색 결과의 상위 N개를 자동으로 선택하는 것으로 대체
print(" Reranker 기능 생략 (LangChain 1.0 호환성)")


# ======================================================
# 4⃣ 하이브리드 검색 함수 (키워드 + 벡터)
# ======================================================
def hybrid_search(query: str, k: int = 2) -> List[Document]:  #  기본값 2개로 변경
    """
    🎯 하이브리드 검색: 키워드 매칭 + 벡터 유사도 + Reranker
    
    단계:
    1. 키워드 추출 (2글자 이상)
    2. 키워드가 포함된 문서 필터링
    3. 필터링된 문서에 대해 벡터 유사도 계산
    4. Reranker로 최종 순위 결정
    5. 키워드 매칭 실패 시 순수 벡터 검색으로 폴백
    """
    print(f"    하이브리드 검색 시작...")
    
    # 1⃣ 키워드 추출
    keywords = [word for word in query.split() if len(word) >= 2]
    print(f"    추출된 키워드: {keywords}")
    
    # 2⃣ DB에서 모든 문서 가져오기
    collection = guideline_db._collection
    all_result = collection.get(include=['documents', 'metadatas'])
    all_contents = all_result.get('documents', [])
    all_metadatas = all_result.get('metadatas', [])
    
    # 3⃣ 키워드 매칭 점수 계산
    keyword_matches = []
    for i, content in enumerate(all_contents):
        score = sum(1 for kw in keywords if kw.lower() in content.lower())
        
        if score > 0:
            keyword_matches.append({
                'content': content,
                'metadata': all_metadatas[i] if i < len(all_metadatas) else {},
                'score': score
            })
    
    print(f"    키워드 매칭 문서: {len(keyword_matches)}개")
    
    # 4⃣ 키워드 매칭된 문서 처리
    if len(keyword_matches) > 0:
        # 점수순 정렬 후 상위 50개
        keyword_matches.sort(key=lambda x: x['score'], reverse=True)
        top_matches = keyword_matches[:min(50, len(keyword_matches))]
        
        # Document 객체로 변환
        candidate_docs = [
            Document(
                page_content=match['content'],
                metadata=match['metadata']
            )
            for match in top_matches
        ]
        
        # 상위 k개 선정 (Reranker 없이)
        print(f"    상위 {k}개 선정...")
        return candidate_docs[:k]
    
    # 5⃣ 키워드 매칭 실패 → 순수 벡터 검색
    print(f"    키워드 매칭 0개, 벡터 검색으로 폴백")
    vector_results = guideline_db.similarity_search(query, k=k)
    return vector_results


# ======================================================
# 5⃣ GuidelineDB 검색 도구 (하이브리드 방식)
# ======================================================
@tool
def guideline_search(query: str) -> List[Document]:
    """
    GuidelineDB에서 하이브리드 검색합니다.
    키워드 매칭 + 벡터 유사도 + Reranker 조합
    """
    print(f"\n [GuidelineDB Hybrid Search] 쿼리: {query}")
    
    # 하이브리드 검색 실행 (상위 2개만)
    docs = hybrid_search(query, k=2)  #  5개 → 2개로 변경
    
    if len(docs) == 0:
        print("❗ 검색 결과 없음")
        return [Document(page_content="관련 정보를 찾을 수 없습니다.", metadata={"source": "guidelineDB"})]
    
    print(f"📄 최종 검색 결과: {len(docs)}개")
    
    # 결과 포맷팅
    formatted_docs = []
    for i, d in enumerate(docs, 1):
        q = d.page_content.strip()
        a = d.metadata.get("answer", "").strip()
        src_detail = d.metadata.get("source_detail", "출처 미기재")
        
        print(f"   ▶ [{i}] {q[:60]}... (출처: {src_detail})")
        
        formatted_docs.append(
            Document(
                page_content=f"Q: {q}\nA: {a}",
                metadata={
                    "source": "guidelineDB",
                    "source_name": "GuidelineDB",
                    "source_detail": src_detail
                }
            )
        )
    
    print(" 검색 완료")
    return formatted_docs


# ======================================================
# 6⃣ 웹 검색 도구
# ======================================================
print(" Tavily Web Search Retriever 초기화 중...")
web_retriever = TavilySearchAPIRetriever(k=10)
print(" Web Retriever 준비 완료.")


@tool
def web_search(query: str) -> List[Document]:
    """
    데이터베이스에 없는 정보 또는 최신 정보를 웹에서 검색합니다.
    (검색된 문서의 제목, URL, 내용 요약, 출처를 포함하여 반환)
    """
    print(f"\n [Web Search] 쿼리 실행: {query}")
    # 검색 실행
    docs = web_retriever.invoke(query)
    
    # 상위 2개 문서만 선별 (Reranker 대신)
    if len(docs) > 2:
        docs = docs[:2]
    formatted_docs = []

    if len(docs) == 0:
        print("❗ 웹 검색 결과 없음")
        return [Document(page_content="관련 정보를 찾을 수 없습니다.", metadata={"source": "web search"})]

    print(f"📄 검색된 문서 수: {len(docs)}")

    for i, doc in enumerate(docs):
        # 안전하게 URL과 제목 추출
        source_url = doc.metadata.get("source", "URL 미기재")
        title = doc.metadata.get("title", "제목 없음")
        snippet = doc.page_content[:400]

        print(f"   [{i+1}] {title}")
        print(f"       -> URL: {source_url}")

        formatted_docs.append(
            Document(
                page_content=(
                    f"🔹 제목: {title}\n"
                    f"🔗 출처 URL: {source_url}\n"
                    f"📄 내용 요약: {snippet}"
                ),
                metadata={
                    "source": "web search",
                    "source_name": title,
                    "source_url": source_url,
                    "source_detail": source_url
                }
            )
        )

    print(f" 웹 검색 결과 {len(formatted_docs)}개 포맷 완료.")
    return formatted_docs


# Cell 12
# 도구 목록을 정의 
tools = [guideline_search, web_search]
