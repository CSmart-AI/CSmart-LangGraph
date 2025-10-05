# Cell 19
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal, List, Dict
from pprint import pprint
from IPython.display import Image, display
import datetime
import re
from step2_states import QAState
from step3_db_and_search import guideline_search
from step4_llm import llm


# ======================================
# 1️⃣ GuidelineRagState 정의
# ======================================
class GuidelineRagState(QAState):
    """
    완성편입 전용 Guideline 검색용 상태 구조
    QAState를 확장하여 추가 필드 포함
    """
    rewritten_query: str              # 재작성된 검색 쿼리
    related_info: List[Dict]          # 추출된 정보 리스트
    node_answer: str                  # 최종 노드 답변
    num_generations: int              # 루프 반복 횟수
    sources: List[str]                # 최종 출처 리스트


# ======================================
# 2️⃣ 공통 로그 함수
# ======================================
def log(message: str, state: Dict = None):
    """시간 + 단계 + 상태 키 표시용 공용 로그 함수"""
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
    if state:
        keys = ', '.join(list(state.keys()))
        print(f"  ↳ 현재 state keys: [{keys}]")


# ======================================
# 3️⃣ Guideline 문서 검색 단계
# ======================================
def retrieve_guideline_docs(state: GuidelineRagState) -> GuidelineRagState:
    """
    GuidelineDB의 [question] 컬럼을 기준으로 Embedding 검색 수행
    """
    log("==== [1단계] retrieve_guideline_docs (question을 기준으로 GuidelineDB에서 검색 수행) 시작 ====", state)

    query = state.get("rewritten_query", state["question"])
    print(f"📢 검색 쿼리 입력값: {query}")
    print("📚 검색 기준 필드: GuidelineDB의 [question] 컬럼 (임베딩 매칭)")

    docs = guideline_search.invoke(query)
    print(f"📄 검색 결과 문서 수: {len(docs)}")

    if len(docs) > 0:
        print("🧾 Top 1 검색 결과 미리보기:")
        preview_q = docs[0].page_content.split("\n")[0][:100]
        src_detail = docs[0].metadata.get("source_detail", "출처 미기재")
        print(f"   ▶ Q: {preview_q}")
        print(f"   ▶ 출처: {src_detail}")
    else:
        print("❗ 검색 결과가 없습니다. 쿼리를 재작성하거나 DB를 확인하세요.")

    # 출처 목록을 sources 필드에 저장
    sources = [
        f"{doc.metadata.get('source_name', 'GuidelineDB')} ({doc.metadata.get('source_detail', '출처 미기재')})"
        for doc in docs
    ]

    return {"search_results": docs, "sources": sources}


# ======================================
# 4️⃣ 문서 정보 추출 및 평가 단계
# ======================================
def extract_guideline_info(state: GuidelineRagState) -> GuidelineRagState:
    log("==== [2단계] extract_guideline_info (문서에서 관련 핵심정보 추출 및 평가) 시작 ====", state)
    extracted_list = []

    try:
        for i, doc in enumerate(state["search_results"]):
            print(f"\n🧾 {i+1}번째 문서 분석 중...")
            src_detail = doc.metadata.get("source_detail", "출처 미기재")
            print(f"   ▶ 출처: {src_detail}")

            doc_q = doc.page_content
            doc_a = doc.metadata.get("answer", "")
            print(f"   Q: {doc_q[:100]}")
            print(f"   A: {doc_a[:100]}")

            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 대학 편입 모집요강 전문가입니다.
                아래 Q/A 문서에서 학생 질문과 관련된 주요 사실을 3~5개 정도 추출하세요. 
                각 항목은 다음과 같은 형식을 따릅니다:

                1. [추출된 정보 요약]
                - 질문과 답변의 관련성: 0~1 사이 숫자
                - 충실성 점수: 0~1 사이 숫자
                """),
                ("human", "질문: {question}\n\n[문서]\nQ: {q}\nA: {a}")
            ])
            formatted = prompt.format(question=state["question"], q=doc_q, a=doc_a)
            result = llm.invoke(formatted)

            if not result or not result.content.strip():
                print("⚠️ LLM 결과 없음 → 문서 스킵")
                continue

            # --- 점수 추출 ---
            text = result.content.strip()
            relevance_scores = [float(x) for x in re.findall(r"관련성\s*점수\s*[:：]?\s*([0-9]*\.?[0-9]+)", text)]
            faithfulness_scores = [float(x) for x in re.findall(r"충실성\s*점수\s*[:：]?\s*([0-9]*\.?[0-9]+)", text)]

            avg_rel = sum(relevance_scores)/len(relevance_scores) if relevance_scores else 0
            avg_fai = sum(faithfulness_scores)/len(faithfulness_scores) if faithfulness_scores else 0

            print(f"   📊 질문과 관련성: {avg_rel:.2f}, 충실성 점수: {avg_fai:.2f}")

            # --- 점수 기준 필터링 ---
            if avg_rel < 0.7 or avg_fai < 0.7:
                print("🚫 점수가 낮아 제외됨 (기준: 0.7 이상)")
                continue

            extracted_list.append({
                "content": text,
                "source": src_detail,
                "avg_relevance": avg_rel,
                "avg_faithfulness": avg_fai
            })

        if len(extracted_list) == 0:
            print("❗ 관련 정보가 추출되지 않았거나 점수 기준 미달입니다.")

        log("✅ 정보 추출 및 필터링 완료", {"추출된 정보 개수": len(extracted_list)})
        return {
            "related_info": extracted_list,
            "num_generations": state.get("num_generations", 0) + 1
        }

    except Exception as e:
        print(f"❌ [오류] extract_guideline_info 실패: {e}")
        return {"related_info": [], "num_generations": 0}


# ======================================
# 5️⃣ 검색 쿼리 재작성 단계
# ======================================
def rewrite_guideline_query(state: GuidelineRagState) -> GuidelineRagState:
    """
    정보 부족 시, LLM을 통해 검색 쿼리 재작성 수행
    """
    log("==== [3단계] rewrite_guideline_query (검색 쿼리 재작성) 시작 ====", state)
    info_text = "\n".join([i["content"] for i in state["related_info"]])

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 대학 편입 전문 상담가입니다.
        아래 질문과 추출된 정보를 참고하여 더 구체적이고 정확한 검색 쿼리를 다시 작성하세요.
        - 핵심 키워드: 학교명, 학과명, 지원유형(일반/학사), 과목, 일정
        - 한 줄로 간결하게 작성
        """),
        ("human", "질문: {question}\n\n추출된 정보:\n{info}")
    ])

    rewritten = llm.invoke(rewrite_prompt.format(question=state["question"], info=info_text))
    new_query = rewritten.content.strip()

    print(f"💡 재작성된 쿼리: {new_query}")
    return {"rewritten_query": new_query}


# ======================================
# 6️⃣ 최종 답변 생성 단계
# ======================================
def generate_guideline_answer(state: GuidelineRagState) -> GuidelineRagState:
    """
    모든 추출 정보를 종합해 학생 질문에 대한 최종 답변 생성
    """
    log("==== [4단계] generate_guideline_answer (최종 답변 생성) 시작 ====", state)

    # 정보 병합 및 출처 표시
    info_text = "\n".join([f"- {i['content']} (출처: {i['source']})" for i in state["related_info"]])
    source_summary = "\n".join([f"- {s}" for s in state.get("sources", [])])

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 대학 편입 모집요강 전문 상담가입니다.
        학생의 질문과 관련 정보를 종합하여 답변을 작성하세요.
        답변은 마크다운 형식으로 작성하며, 각 정보의 출처를 명확히 표시해야 합니다.
        출력 구조:
        1. 핵심 요약
        2. 세부 내용
        3. 참고 출처
        """),
        ("human", "질문: {question}\n\n관련 정보:\n{info}\n\n참고 출처:\n{src}")
    ])

    answer = llm.invoke(answer_prompt.format(
        question=state["question"],
        info=info_text,
        src=source_summary
    ))

    print("🗒️ 생성된 답변 미리보기:\n", answer.content[:300], "...")
    log("✅ 최종 답변 생성 완료")
    return {"node_answer": answer.content, "sources": state.get("sources", [])}


# ======================================
# 7️⃣ 판단 단계
# ======================================
def should_continue_guideline(state: GuidelineRagState) -> Literal["계속", "종료"]:
    """
    정보 충분 여부에 따라 그래프 진행 방향 결정
    """
    log("==== [판단단계] should_continue_guideline (정보 충족 여부 판단) ====", state)
    print("📊 현재 related_info 개수:", len(state.get("related_info", [])))
    print("🔢 현재 반복 횟수:", state.get("num_generations", 0))

    gen = state.get("num_generations", 0)
    info_count = len(state.get("related_info", []))

    if gen >= 2:
        print("🔁 반복 횟수 초과 → 종료")
        return "종료"

    if info_count > 0:
        print(f"📈 충분한 정보 확보 ({info_count}개) → 종료")
        return "종료"

    print("🔄 정보 부족 → 쿼리 재작성 후 재검색")
    return "계속"


# ======================================
# 8️⃣ 그래프 구성 및 컴파일
# ======================================
log("🚀 [초기화] LangGraph Guideline Search Workflow 구성 시작")

guideline_graph = StateGraph(GuidelineRagState)

guideline_graph.add_node("retrieve", retrieve_guideline_docs)
guideline_graph.add_node("extract", extract_guideline_info)
guideline_graph.add_node("rewrite", rewrite_guideline_query)
guideline_graph.add_node("answer", generate_guideline_answer)

guideline_graph.add_edge(START, "retrieve")
guideline_graph.add_edge("retrieve", "extract")

guideline_graph.add_conditional_edges(
    "extract",
    should_continue_guideline,
    {"계속": "rewrite", "종료": "answer"}
)

guideline_graph.add_edge("rewrite", "retrieve")
guideline_graph.add_edge("answer", END)

guideline_agent = guideline_graph.compile()
log("✅ [완료] Guideline Agent 컴파일 완료")


# ======================================
# 🎯 완성편입 Guideline Graph (표준 노드명 버전)
# ======================================

# 그래프 생성
workflow = StateGraph(GuidelineRagState)

# 노드 추가 (표준화된 이름 사용)
workflow.add_node("retrieve_documents", retrieve_guideline_docs)       # GuidelineDB 검색
workflow.add_node("extract_and_evaluate", extract_guideline_info)      # 정보 추출 및 점수 평가
workflow.add_node("rewrite_query", rewrite_guideline_query)            # 검색 쿼리 재작성
workflow.add_node("generate_answer", generate_guideline_answer)        # 최종 답변 생성

# 엣지 연결
workflow.add_edge(START, "retrieve_documents")
workflow.add_edge("retrieve_documents", "extract_and_evaluate")

# 조건부 엣지 연결
workflow.add_conditional_edges(
    "extract_and_evaluate",
    should_continue_guideline,  # 판단 로직
    {
        "계속": "rewrite_query",
        "종료": "generate_answer"
    }
)

# 루프: 쿼리 재작성 후 다시 검색
workflow.add_edge("rewrite_query", "retrieve_documents")

# 최종 답변 후 종료
workflow.add_edge("generate_answer", END)

# 그래프 컴파일
guideline_agent = workflow.compile()

# ======================================
# 🧭 그래프 시각화
# ======================================
display(Image(guideline_agent.get_graph().draw_mermaid_png()))
print("\n✅ [완료] Guideline Agent 그래프 구조 시각화 완료 ✅")

# ======================================
# 🌐 트리 구조 텍스트 로그
# ======================================
print("\n📊 Guideline Workflow 구조:")
print("""
START → retrieve_documents → extract_and_evaluate
 ├─(계속)→ rewrite_query → retrieve_documents
 └─(종료)→ generate_answer → END
""")
