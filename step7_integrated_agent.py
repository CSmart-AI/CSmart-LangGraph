# Cell 25-27
from typing import Annotated, List, Dict
from operator import add
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
try:
    from IPython.display import Image, display
except ImportError:
    # IPython이 없으면 무시 (노트북 환경이 아닌 경우)
    Image = None
    display = None
from typing import Literal
from step4_llm import llm
from step5_guideline_agent import guideline_agent
from step6_web_agent import search_web_agent

# ======================================
# 통합 에이전트 상태 정의 ( prepare_context 활용)
# ======================================
class IntegratedAgentState(TypedDict):
    question: str
    student_profile: Dict[str, str]           #  학생 프로필
    recent_dialogues: List[Dict[str, str]]    #  최근 대화 내역
    context: str                              #  prepare_context로 생성
    answers: Annotated[List[str], add]
    final_answer: str
    datasources: List[str]

# ======================================
# 라우팅 결정을 위한 데이터 모델
# ======================================
class ToolSelector(BaseModel):
    """Routes the user question to the most appropriate tool."""
    tool: Literal["search_guideline", "search_web"] = Field(
        description="Select one of the tools, based on the user's question.",
    )

class ToolSelectors(BaseModel):
    """Select the appropriate tools that are suitable for the user question."""
    tools: List[ToolSelector] = Field(
        description="Select one or more tools, based on the user's question.",
    )

# 구조화된 출력을 위한 LLM 설정
structured_llm_tool_selector = llm.with_structured_output(ToolSelectors)

# 라우팅을 위한 프롬프트 템플릿
system = """당신은 대학 편입 상담 전문 AI 어시스턴트입니다. 
다음 가이드라인에 따라 사용자 질문을 적절한 도구로 라우팅하세요:

- 특정 대학의 편입 모집요강, 시험 과목, 전형 방법 등 내부 가이드라인DB에 있을만한 질문은 search_guideline 도구를 사용하세요.
- 최신 정보, 입시 일정, 합격자 발표, 또는 가이드라인DB에 없을 것 같은 정보는 search_web 도구를 사용하세요.
- 질문이 애매하거나 두 가지 모두 필요한 경우, 두 도구를 모두 선택하세요.

항상 사용자 질문에 가장 적합한 도구를 선택하세요."""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

# 질문 라우터 정의
question_tool_router = route_prompt | structured_llm_tool_selector

print(" 질문 라우터 설정 완료")


# ======================================
# 컨텍스트 준비 노드 ( 원래 prepare_context 함수 재사용)
# ======================================
def prepare_context_node(state: IntegratedAgentState) -> IntegratedAgentState:
    """
    학생 프로필과 최근 대화 내역을 종합하여 context 생성
    (Cell 6의 원래 prepare_context 로직 기반)
    """
    print("\n --- 컨텍스트 준비 중 ---")
    
    # 학생 프로필 불러오기
    profile = state.get("student_profile", {})
    target_uni = profile.get("target_university", "미지정")
    # 'track' 또는 'major_category' 모두 지원
    track = profile.get("track", profile.get("major_category", "계열 미지정"))
    
    # 최근 대화 내역 가져오기 (학생과 선생님 5개 정도)
    dialogues = state.get("recent_dialogues", [])
    dialogue_summary = " ".join(
        [f"{d['role']}: {d['message']}" for d in dialogues[-5:]]
    )
    
    # 질문과 맥락 결합
    context = (
        f"[학생 프로필] 목표 대학: {target_uni}, 계열: {track}\n"
        f"[최근 대화 요약] {dialogue_summary}\n"
        f"[학생 질문] {state['question']}"
    )
    
    print(f" 컨텍스트 생성 완료")
    print(f"   - 목표 대학: {target_uni}")
    print(f"   - 계열: {track}")
    print(f"   - 대화 내역: {len(dialogues)}개")
    
    return {"context": context}


# ======================================
# 질문 라우팅 노드 정의
# ======================================
def analyze_question_tool_search(state: IntegratedAgentState):
    """사용자 질문을 분석하여 적절한 도구를 선택"""
    question = state["question"]
    context = state.get("context", "")
    
    print(f"\n 질문 분석 중: {question}")
    
    # 컨텍스트 포함하여 분석 (더 정확한 라우팅)
    query = f"{context}\n\n질문: {question}" if context else question
    result = question_tool_router.invoke({"question": query})
    datasources = [tool.tool for tool in result.tools]
    print(f" 선택된 도구: {datasources}")
    return {"datasources": datasources}


def route_datasources_tool_search(state: IntegratedAgentState) -> List[str]:
    """선택된 데이터 소스에 따라 라우팅"""
    datasources = set(state['datasources'])
    valid_sources = {"search_guideline", "search_web"}
    
    if datasources.issubset(valid_sources):
        return list(datasources)
    
    return list(valid_sources)


# ======================================
# 서브 에이전트 노드 정의 ( 컨텍스트 활용)
# ======================================
def guideline_rag_node(state: IntegratedAgentState) -> IntegratedAgentState:
    """GuidelineDB 검색 에이전트 실행 (컨텍스트 포함)"""
    print("\n---  GuidelineDB 검색 에이전트 시작 ---")
    question = state["question"]
    context = state.get("context", "")
    
    try:
        # 컨텍스트와 함께 질문 전달
        enriched_question = f"{context}\n\n질문: {question}" if context else question
        
        # 타임아웃 설정 및 안전한 호출
        answer = guideline_agent.invoke(
            {"question": enriched_question},
            config={"recursion_limit": 10}  #  재귀 제한
        )
        print(" GuidelineDB 검색 완료")
        
        # 안전하게 답변 추출
        node_answer = answer.get("node_answer", "")
        if not node_answer:
            node_answer = "GuidelineDB에서 관련 정보를 찾을 수 없습니다."
        else:
            # 출처 정보 추가
            node_answer = f"[GuidelineDB 검색 결과]\n{node_answer}"
            
        return {"answers": [node_answer]}
        
    except Exception as e:
        print(f" GuidelineDB 검색 오류: {str(e)[:100]}")
        return {"answers": ["GuidelineDB 검색 중 오류가 발생했습니다."]}


def web_rag_node(state: IntegratedAgentState) -> IntegratedAgentState:
    """웹 검색 에이전트 실행 (컨텍스트 포함)"""
    print("\n---  웹 검색 에이전트 시작 ---")
    question = state["question"]
    context = state.get("context", "")
    
    try:
        # 컨텍스트와 함께 질문 전달
        enriched_question = f"{context}\n\n질문: {question}" if context else question
        
        # 타임아웃 설정 및 안전한 호출
        answer = search_web_agent.invoke(
            {"question": enriched_question},
            config={"recursion_limit": 10}  #  재귀 제한
        )
        print(" 웹 검색 완료")
        
        # 안전하게 답변 추출
        node_answer = answer.get("node_answer", "")
        if not node_answer:
            node_answer = "웹 검색에서 관련 정보를 찾을 수 없습니다."
        else:
            # 출처 정보 추가
            node_answer = f"[웹 검색 결과]\n{node_answer}"
            
        return {"answers": [node_answer]}
        
    except Exception as e:
        print(f" 웹 검색 오류: {str(e)[:100]}")
        return {"answers": ["웹 검색 중 오류가 발생했습니다."]}


# ======================================
# 최종 답변 생성 노드
# ======================================
# RAG 프롬프트 정의
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 대학 편입 전문 상담가입니다. 제공된 정보를 바탕으로 학생의 질문에 답변하세요.

다음 가이드라인을 따르세요:
1. 제공된 문서의 정보만 사용하세요.
2. 문서에 관련 정보가 부족하면 "제공된 정보로는 충분한 답변을 할 수 없습니다"라고 말하세요.
3. 각 문장마다 정보의 출처를 명시하세요. 형식:
   - GuidelineDB: "(출처 GuidelineDB (출처명))"
   - 웹 검색: "(출처 제목 (URL))"
4. 추측하거나 문서에 없는 정보를 추가하지 마세요.
5. 친절한 선생님의 말투로 학생들에게 직접 과외하듯 답변하되 필요없는 서론은 제외하고 본론만 간결하고 명확하게 작성하세요.
6. 불필요한 정보는 생략하세요.
7. 여러 출처에서 같은 정보를 제공하면 모든 관련 출처를 인용하세요.
8. 여러 출처의 정보를 결합할 때는 각 출처를 인용하면서 일관되게 작성하세요.
9. 어떤 정보를 통해서 답변을 만들거나 AI 툴을 활용하는 것을 알아차리지 못하도록 온전히 사람이 직접 생각해서 답변한 것처럼 해주세요. (제공된 정보에 의하면~ 와 같은 말 사용 금지)
10. 특수문자를 사용하지 말고 사람이 말하듯 한 문단으로 작성해주세요.

출처 인용 예시:
"중앙대학교 이과 편입은 수학 시험만 응시합니다 (GuidelineDB (cau2025-page7)). 시험 일정은 2025년 1월 18일입니다 (중앙대 편입 모집요강, https://example.com)."
"""),
    ("human", "다음 문서를 사용하여 질문에 답변하세요:\n\n[문서]\n{documents}\n\n[질문]\n{question}"),
])


def answer_final(state: IntegratedAgentState) -> IntegratedAgentState:
    """수집된 정보를 종합하여 최종 답변 생성 (컨텍스트 활용)"""
    print("\n---  최종 답변 생성 중 ---")
    question = state["question"]
    context = state.get("context", "")
    documents = state.get("answers", [])
    
    if not isinstance(documents, list):
        documents = [documents]

    # 문서 내용을 문자열로 결합
    documents_text = "\n\n".join(documents)
    
    # 컨텍스트와 함께 최종 질문 생성
    enriched_question = f"{context}\n\n질문: {question}" if context else question

    # RAG generation
    rag_chain = rag_prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({
        "documents": documents_text, 
        "question": enriched_question
    })
    print(" 최종 답변 생성 완료")
    return {"final_answer": generation, "question": question}


print(" 노드 정의 완료")


# ======================================
# 통합 그래프 구성 ( prepare_context 포함)
# ======================================
# LangGraph의 StateGraph, START, END 노드 임포트

# ======================================
# 노드 정의를 딕셔너리로 관리
# - 노드 이름(key)과 실행 함수(value)를 매핑
# ======================================
nodes = {
    "prepare_context": prepare_context_node,           # 컨텍스트 준비: 학생 프로필 + 대화 내역 → context 생성
    "analyze_question": analyze_question_tool_search,  # 질문 분석: LLM이 질문을 분석하여 적절한 도구 선택 (GuidelineDB / Web)
    "search_guideline": guideline_rag_node,            # GuidelineDB 검색: 내부 DB에서 편입 정보 검색
    "search_web": web_rag_node,                        # 웹 검색: Tavily API로 최신 정보 검색
    "generate_answer": answer_final,                   # 최종 답변 생성: 수집된 정보를 종합하여 답변 생성
}

# ======================================
# 그래프 생성: IntegratedAgentState를 상태로 사용
# - 모든 노드가 이 상태 구조를 공유함
# ======================================
integrated_builder = StateGraph(IntegratedAgentState)

# ======================================
# 노드 추가: 딕셔너리에 정의된 모든 노드를 그래프에 등록
# ======================================
for node_name, node_func in nodes.items():
    integrated_builder.add_node(node_name, node_func)  # 각 노드를 그래프에 추가

# ======================================
# 엣지 추가 (병렬 처리 지원)
# ======================================

# 1⃣ START → prepare_context
# - 워크플로우 시작: 가장 먼저 컨텍스트를 준비
# - 입력: question, student_profile, recent_dialogues
# - 출력: context (형식화된 컨텍스트 문자열)
integrated_builder.add_edge(START, "prepare_context")

# 2⃣ prepare_context → analyze_question
# - 컨텍스트 준비 완료 후 질문 분석 단계로 이동
# - LLM이 컨텍스트와 질문을 분석하여 어떤 도구를 사용할지 결정
integrated_builder.add_edge("prepare_context", "analyze_question")

# 3⃣ analyze_question → 조건부 라우팅 (도구 선택)
# - route_datasources_tool_search 함수가 반환한 도구로 라우팅
# - 반환 가능한 값: ["search_guideline"], ["search_web"], ["search_guideline", "search_web"]
# - 두 개가 선택되면 병렬로 실행됨!
integrated_builder.add_conditional_edges(
    "analyze_question",                      # 출발 노드
    route_datasources_tool_search,           # 라우팅 결정 함수 (어느 노드로 갈지 결정)
    ["search_guideline", "search_web"]       # 가능한 목적지 노드 리스트
)

# 4⃣ 검색 노드들을 generate_answer에 연결
# - search_guideline과 search_web 모두 generate_answer로 연결
# - 병렬 실행 가능: 두 검색이 동시에 진행되고 모두 완료되면 generate_answer 실행
# - answers 필드는 Annotated[List[str], add]로 정의되어 자동으로 병합됨
for node in ["search_guideline", "search_web"]:
    integrated_builder.add_edge(node, "generate_answer")

# 5⃣ generate_answer → END
# - 최종 답변 생성 후 워크플로우 종료
# - 출력: final_answer (학생에게 제공할 최종 답변)
integrated_builder.add_edge("generate_answer", END)

# ======================================
# 그래프 컴파일
# - StateGraph를 실행 가능한 객체로 변환
# - 이후 integrated_agent.invoke()로 실행 가능
# ======================================
integrated_agent = integrated_builder.compile()

# ======================================
# 그래프 시각화
# - Mermaid 다이어그램으로 워크플로우 구조 표시
# ======================================
if display is not None and Image is not None:
    display(Image(integrated_agent.get_graph().draw_mermaid_png()))
print("\n [완료] 통합 에이전트 구성 완료 (prepare_context 포함)")
