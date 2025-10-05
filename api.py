"""
CSmart 편입 상담 API
get_answer() 함수 하나로 모든 기능을 사용할 수 있습니다.

사용법:
    from api import get_answer
    
    result = get_answer("중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
    print(result["final_answer"])
"""

from typing import Dict, List, Optional
from dotenv import load_dotenv
import os

# ======================================
# 1단계: 환경 변수 로드
# ======================================
load_dotenv()

# ======================================
# 2단계: 모든 필요한 모듈 import 및 초기화
# ======================================
print("🚀 CSmart API 초기화 중...")

# States
from step2_states import QAState, prepare_context

# DB and Search (자동으로 초기화됨)
from step3_db_and_search import guideline_search, web_search, tools

# LLM (자동으로 초기화됨)
from step4_llm import llm, llm_with_tools

# Agents (자동으로 컴파일됨)
from step5_guideline_agent import guideline_agent
from step6_web_agent import search_web_agent
from step7_integrated_agent import integrated_agent

print("✅ CSmart API 초기화 완료!\n")


# ======================================
# 🎯 메인 API 함수
# ======================================
def get_answer(
    question: str,
    student_profile: Optional[Dict[str, str]] = None,
    recent_dialogues: Optional[List[Dict[str, str]]] = None,
    verbose: bool = True
) -> Dict:
    """
    편입 상담 질문에 대한 답변을 생성합니다.
    
    Parameters:
    -----------
    question : str
        학생의 질문 (예: "중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
    
    student_profile : dict, optional
        학생 프로필 정보
        예: {"target_university": "중앙대학교", "track": "이과"}
        기본값: {"target_university": "미지정", "track": "계열 미지정"}
    
    recent_dialogues : list, optional
        최근 대화 내역 (최대 5개)
        예: [
            {"role": "student", "message": "..."},
            {"role": "teacher", "message": "..."}
        ]
        기본값: 빈 리스트
    
    verbose : bool, optional
        상세 로그 출력 여부 (기본값: True)
        False로 설정하면 로그를 숨깁니다.
    
    Returns:
    --------
    dict
        {
            "question": str,           # 원본 질문
            "final_answer": str,       # 최종 답변
            "context": str,            # 생성된 컨텍스트
            "datasources": list,       # 사용된 데이터 소스
            "success": bool,           # 성공 여부
            "error": str or None       # 오류 메시지 (있는 경우)
        }
    
    Examples:
    ---------
    >>> from api import get_answer
    >>> 
    >>> # 간단한 사용 (로그 출력됨)
    >>> result = get_answer("중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
    >>> print(result["final_answer"])
    >>> 
    >>> # 프로필과 대화 내역 포함
    >>> result = get_answer(
    ...     question="중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?",
    ...     student_profile={"target_university": "중앙대학교", "track": "이과"},
    ...     recent_dialogues=[
    ...         {"role": "student", "message": "편입 준비를 시작하려고 합니다."},
    ...         {"role": "teacher", "message": "어느 대학을 목표로 하시나요?"}
    ...     ]
    ... )
    >>> print(result["final_answer"])
    >>> 
    >>> # 로그 숨기기
    >>> result = get_answer("질문", verbose=False)
    >>> print(result["final_answer"])
    """
    
    # 기본값 설정
    if student_profile is None:
        student_profile = {
            "target_university": "미지정",
            "track": "계열 미지정"
        }
    
    if recent_dialogues is None:
        recent_dialogues = []
    
    # 입력 데이터 구성
    inputs = {
        "question": question,
        "student_profile": student_profile,
        "recent_dialogues": recent_dialogues
    }
    
    try:
        if not verbose:
            # 로그 출력 최소화
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
        
        # 통합 에이전트 실행
        result = integrated_agent.invoke(
            inputs,
            config={
                "recursion_limit": 25,  # 재귀 제한
                "timeout": 120          # 2분 타임아웃
            }
        )
        
        if not verbose:
            sys.stdout = old_stdout
        
        # 결과 반환
        return {
            "question": question,
            "final_answer": result.get("final_answer", "답변을 생성하지 못했습니다."),
            "context": result.get("context", ""),
            "datasources": result.get("datasources", []),
            "success": True,
            "error": None
        }
        
    except Exception as e:
        if not verbose:
            sys.stdout = old_stdout
        
        error_msg = str(e)
        print(f"❌ 오류 발생: {error_msg[:200]}")
        
        return {
            "question": question,
            "final_answer": "오류로 인해 답변을 생성하지 못했습니다.",
            "context": "",
            "datasources": [],
            "success": False,
            "error": error_msg
        }


# ======================================
# 🧪 API 테스트 (이 파일을 직접 실행할 때)
# ======================================
if __name__ == "__main__":
    print("=" * 80)
    print("CSmart API 테스트")
    print("=" * 80)
    
    # 테스트: 전체 정보 포함 (로그 출력)
    print("\n[테스트] 프로필 및 대화 내역 포함")
    result = get_answer(
        question="중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?",
        student_profile={"target_university": "중앙대학교", "track": "이과"},
        recent_dialogues=[
            {"role": "student", "message": "편입 준비를 시작하려고 합니다."},
            {"role": "teacher", "message": "좋습니다. 어느 대학을 목표로 하시나요?"},
            {"role": "student", "message": "중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?"}
        ]
    )
    
    print("\n" + "=" * 80)
    print("최종 결과:")
    print("=" * 80)
    print(f"질문: {result['question']}")
    print(f"성공 여부: {result['success']}")
    print(f"사용된 데이터 소스: {result['datasources']}")
    print(f"\n답변:\n{result['final_answer']}")
    
    print("\n" + "=" * 80)
    print("✅ API 테스트 완료")
    print("=" * 80)
