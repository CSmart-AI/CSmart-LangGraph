# Cell 22
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal, Optional, List
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from step2_states import QAState
from step3_db_and_search import web_search
from step4_llm import llm

# ==============================
# 0️⃣ Pydantic 스키마 정의 (필수!)
# ==============================
class InformationStrip(BaseModel):
    """추출된 정보 조각"""
    content: str = Field(description="추출된 정보 내용")
    relevance_score: float = Field(description="질문과의 관련성 점수 (0-1)")
    faithfulness_score: float = Field(description="답변의 충실성 점수 (0-1)")
    source: str = Field(default="출처 미상", description="정보 출처 URL")

class ExtractedInformation(BaseModel):
    """문서에서 추출된 전체 정보"""
    strips: List[InformationStrip] = Field(description="추출된 정보 조각 리스트")
    query_relevance: float = Field(description="문서 전체의 질문 관련성 (0-1)")

class RefinedQuestion(BaseModel):
    """재작성된 검색 쿼리"""
    question_refined: str = Field(description="개선된 검색 쿼리")
    reason: str = Field(default="", description="재작성 이유")

# ==============================
# 1️⃣ SearchRagState 정의
# ==============================
class SearchRagState(QAState):
    rewritten_query: Optional[str] = None          # 재작성한 질문
    documents: Optional[List] = None               # 검색된 문서 리스트 ✅ 추가!
    extracted_info: Optional[List] = None          # 추출된 정보 조각 리스트
    node_answer: Optional[str] = None              # 최종 답변
    num_generations: int = 0                       # 반복 횟수


# ==============================
# 2️⃣ 문서 검색 단계
# ==============================
def retrieve_documents(state: SearchRagState) -> SearchRagState:
    print("🌐 --- [1단계] 문서 검색 ---")
    query = state.get("rewritten_query", state["question"])
    print(f"🔎 검색 쿼리: {query}")
    docs = web_search.invoke(query)
    print(f"📄 검색 결과 문서 수: {len(docs)}")
    return {"documents": docs}


# ==============================
# 3️⃣ 정보 추출 및 평가 단계
# ==============================
def extract_and_evaluate_information(state: SearchRagState) -> SearchRagState:
    print("🧩 --- [2단계] 정보 추출 및 평가 ---")

    # ✅ 안전하게 documents 가져오기
    docs = state.get("documents", [])
    if not docs:
        print("❗ 문서가 없습니다.")
        return {"extracted_info": [], "num_generations": state.get("num_generations", 0) + 1}

    extracted_strips = []
    MAX_DOC_LENGTH = 3000  # 🔒 문서 최대 길이 제한 (메모리 보호)

    for idx, doc in enumerate(docs[:3]):  # 🔒 최대 3개 문서만 처리
        print(f"\n📘 문서 {idx+1}/{len(docs[:3])} 분석 중...")
        
        try:
            # 🔒 문서 내용 길이 제한 (메모리 과부하 방지)
            doc_content = doc.page_content[:MAX_DOC_LENGTH]
            if len(doc.page_content) > MAX_DOC_LENGTH:
                print(f"⚠️ 문서가 너무 큽니다. {MAX_DOC_LENGTH}자로 자름")
            
            extract_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 인터넷 정보 검색 전문가입니다. 주어진 문서에서 질문과 관련된 주요 사실과 정보를 최대 3개만 간결하게 추출하세요. 
                각 추출된 정보에 대해 다음 두 가지 측면을 0에서 1 사이의 점수로 평가하세요:
                1. 질문과 답변의 관련성
                2. 답변의 충실성
                
                마지막으로, 문서 전체의 질문 관련성을 0에서 1 사이의 점수로 평가하세요."""),
                ("human", "[질문]\n{question}\n\n[문서 내용]\n{document_content}")
            ])

            extract_llm = llm.with_structured_output(ExtractedInformation)
            
            extracted_data = extract_llm.invoke(extract_prompt.format(
                question=state["question"],
                document_content=doc_content
            ))

            print(f"   📊 문서 관련성: {extracted_data.query_relevance:.2f}")
            
            if extracted_data.query_relevance < 0.7:  # 기준 완화 (0.8 → 0.7)
                print("   ⚠️ 문서 관련성 낮음 → 제외")
                continue

            for strip in extracted_data.strips:
                if strip.relevance_score >= 0.7 and strip.faithfulness_score >= 0.7:
                    strip.source = doc.metadata.get("source_url", doc.metadata.get("url", "출처 미상"))
                    extracted_strips.append(strip)
                    print(f"   ✅ 정보 추출: {strip.content[:50]}...")
        
        except Exception as e:
            print(f"   ❌ 문서 처리 오류: {str(e)[:100]}")
            continue  # 오류 발생 시 다음 문서로

    print(f"\n✅ 총 추출된 정보 개수: {len(extracted_strips)}")

    return {
        "extracted_info": extracted_strips,
        "num_generations": state.get("num_generations", 0) + 1
    }


# ==============================
# 4️⃣ 쿼리 재작성 단계
# ==============================
def rewrite_query(state: SearchRagState) -> SearchRagState:
    print("🪄 --- [3단계] 쿼리 재작성 ---")

    extracted_info_str = "\n".join([strip.content for strip in state.get("extracted_info", [])])

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 인터넷 정보 검색 전문가입니다. 주어진 원래 질문과 추출된 정보를 바탕으로, 더 관련성 있고 충실한 정보를 찾기 위해 검색 쿼리를 개선해주세요.

        다음 사항을 고려하여 검색 쿼리를 개선하세요:
        1. 원래 질문의 핵심 요소
        2. 추출된 정보의 관련성 점수
        3. 추출된 정보의 충실성 점수
        4. 부족한 정보나 더 자세히 알아야 할 부분

        개선된 검색 쿼리 작성 단계:
        1. 2-3개의 검색 쿼리를 제안하세요.
        2. 각 쿼리는 구체적이고 간결해야 합니다(5-10 단어 사이).
        3. 질문과 관련된 전문 용어를 적절히 활용하세요.
        4. 각 쿼리 뒤에는 해당 쿼리를 제안한 이유를 간단히 설명하세요.

        출력 형식:
        1. [개선된 검색 쿼리 1]
        - 이유: [이 쿼리를 제안한 이유 설명]
        2. [개선된 검색 쿼리 2]
        - 이유: [이 쿼리를 제안한 이유 설명]
        3. [개선된 검색 쿼리 3]
        - 이유: [이 쿼리를 제안한 이유 설명]

        마지막으로, 제안된 쿼리 중 가장 효과적일 것 같은 쿼리를 선택하고 그 이유를 설명하세요."""),
        ("human", "질문: {question}\n\n추출된 정보:\n{extracted_info}")
    ])

    rewrite_llm = llm.with_structured_output(RefinedQuestion)
    response = rewrite_llm.invoke(rewrite_prompt.format(
        question=state["question"],
        extracted_info=extracted_info_str
    ))

    print(f"💡 재작성된 쿼리: {response.question_refined}")
    return {"rewritten_query": response.question_refined}


# ==============================
# 5️⃣ 최종 답변 생성 단계
# ==============================
def generate_node_answer(state: SearchRagState) -> SearchRagState:
    print("🧠 --- [4단계] 답변 생성 ---")

    extracted_info_str = "\n".join([
        f"- {strip.content} (출처: {strip.source}, 관련성: {strip.relevance_score:.2f}, 충실성: {strip.faithfulness_score:.2f})"
        for strip in state.get("extracted_info", [])
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 인터넷 정보 검색 전문가입니다. 주어진 질문과 추출된 정보를 바탕으로 답변을 생성해주세요. 
        답변은 마크다운 형식으로 작성하며, 각 정보의 출처를 명확히 표시해야 합니다. 
        답변 구조:
        1. 질문에 대한 직접적인 답변
        2. 관련 출처 및 링크
        3. 추가 설명 또는 예시 (필요한 경우)
        4. 결론 및 요약
        각 섹션에서 사용된 정보의 출처를 괄호 안에 명시하세요. 예: (출처: 블로그 (www.blog.com/page/001)"""),
        ("human", "질문: {question}\n\n추출된 정보:\n{extracted_info}")
    ])

    node_answer = llm.invoke(answer_prompt.format(
        question=state["question"],
        extracted_info=extracted_info_str
    ))

    print("📝 생성된 답변 미리보기:\n", node_answer.content[:300], "...")
    return {"node_answer": node_answer.content}


# ==============================
# 6️⃣ 반복 판단 단계
# ==============================
def should_continue(state: SearchRagState) -> Literal["계속", "종료"]:
    if state["num_generations"] >= 2:
        print("🔁 반복 횟수 초과 → 종료")
        return "종료"
    if len(state.get("extracted_info", [])) >= 1:
        print("✅ 충분한 정보 확보 → 종료")
        return "종료"
    print("🔄 정보 부족 → 쿼리 재작성 후 재검색")
    return "계속"


# ==============================
# 7️⃣ LangGraph 구성
# ==============================
workflow = StateGraph(SearchRagState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("extract_and_evaluate", extract_and_evaluate_information)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_answer", generate_node_answer)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "extract_and_evaluate")

workflow.add_conditional_edges(
    "extract_and_evaluate",
    should_continue,
    {"계속": "rewrite_query", "종료": "generate_answer"}
)

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate_answer", END)

# 컴파일 및 시각화
search_web_agent = workflow.compile()
display(Image(search_web_agent.get_graph().draw_mermaid_png()))
print("\n✅ [완료] 웹 검색 기반 RAG 에이전트 구성 완료")
