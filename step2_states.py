# Cell 4
from typing import TypedDict, List, Dict

# ======================================
# LangGraph 상태 구조 정의 (QAState)
# ======================================
# TypedDict: Python의 타입 힌트를 위한 딕셔너리 구조
# - LangGraph의 모든 노드가 이 상태 구조를 공유하며 데이터를 전달함
# - 각 노드는 필요한 필드를 읽고, 업데이트된 필드를 반환함
# ======================================
class QAState(TypedDict):
    # 1️⃣ 사용자 입력 정보
    question: str                           # 학생의 질문 (예: "중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
    student_profile: Dict[str, str]         # 학생 프로필 정보 (예: {"target_university": "중앙대학교", "track": "이과"})
    recent_dialogues: List[Dict[str, str]]  # 최근 대화 내역 (예: [{"role": "student", "message": "..."}, {"role": "teacher", "message": "..."}])
    
    # 2️⃣ 컨텍스트 (prepare_context 노드에서 생성)
    context: str                            # 학생 프로필 + 대화 내역 + 질문을 종합한 컨텍스트 문자열
    
    # 3️⃣ 질문 분류 및 FAQ (현재 통합 에이전트에서는 미사용)
    category: str                           # 질문 카테고리 (예: "입시일정", "시험과목", "모집요강")
    faq_answer: str                         # FAQ 검색 결과 (자주 묻는 질문에 대한 사전 답변)
    
    # 4️⃣ 검색 결과
    search_results: List[Dict]              # 검색된 문서 리스트 (RAG 검색 결과)
    
    # 5️⃣ 답변 생성
    candidate_answer: str                   # 후보 답변 (최종 답변 전 임시 답변)
    evaluation: str                         # 답변 평가 결과 (품질 체크용)
    final_answer: str                       # 최종 답변 (학생에게 제공할 최종 답변)
    
    # 6️⃣ 출처 정보
    sources: List[str]                      # 답변의 출처 리스트 (문서 ID, URL 등)


# Cell 5
def prepare_context(state: QAState) -> QAState:
    """
    학생 프로필과 최근 대화 내역을 종합하여 context 생성
    """
    # 학생 프로필 불러오기
    profile = state.get("student_profile", {})
    target_uni = profile.get("target_university", "미지정")
    track = profile.get("track", "계열 미지정")

    # 최근 대화 내역 가져오기 (학생과 선생님 5개 정도)
    dialogues = state.get("recent_dialogues", [])
    dialogue_summary = " ".join(
        [f"{d['role']}: {d['message']}" for d in dialogues[-5:]]
    )

    # 질문과 맥락 결합
    state["context"] = (
        f"[학생 프로필] 목표 대학: {target_uni}, 계열: {track}\n"
        f"[최근 대화 요약] {dialogue_summary}\n"
        f"[학생 질문] {state['question']}"
    )

    return state
