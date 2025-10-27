"""
get_answer() 함수 하나로 모든 기능을 사용할 수 있습니다.

사용법:
    from api import get_answer
    
    result = get_answer("중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
    print(result["final_answer"])
"""

from typing import Dict, List, Optional
from dotenv import load_dotenv
import os
import requests
from pydantic import BaseModel
from typing import Literal

# ======================================
# 1단계: 환경 변수 로드
# ======================================
load_dotenv()

# ======================================
# 2단계: 모든 필요한 모듈 import 및 초기화
# ======================================
print("CSmart API 초기화 중...")

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

print("CSmart API 초기화 완료!\n")


# ======================================
# 🎯 질문 복잡도 판별을 위한 데이터 모델
# ======================================
class QuestionComplexity(BaseModel):
    """질문의 복잡도를 판별합니다."""
    complexity: Literal["simple", "complex"] = None
    reason: str = None


# ======================================
# 🎯 파인튜닝 모델 답변 품질 평가를 위한 데이터 모델
# ======================================
class AnswerQuality(BaseModel):
    """파인튜닝 모델 답변의 품질을 평가합니다."""
    quality: Literal["good", "poor"] = None
    reason: str = None
    score: int = None  # 1-10 점수


# ======================================
# 🤖 간단한 질문 판별 함수
# ======================================
def is_simple_question(question: str, verbose: bool = True) -> bool:
    """
    질문이 간단한 일반적인 학습 조언인지, 
    특정 대학/일정/전형 정보가 필요한 복잡한 질문인지 판별합니다.
    
    간단한 질문 예시:
    - "수학 공부는 어떻게 해야 할까요?"
    - "오답노트는 어떻게 정리할까요?"
    - "영어 단어 암기는 어떻게 해야 할까요?"
    
    복잡한 질문 예시:
    - "중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?" (특정 대학/계열)
    - "2025학년도 편입 시험 일정은 언제인가요?" (구체적인 날짜 정보)
    """
    from langchain_core.prompts import ChatPromptTemplate
    
    try:
        # 구조화된 출력을 위한 LLM 설정
        structured_llm = llm.with_structured_output(QuestionComplexity)
        
        # 판별 프롬프트
        system_prompt = """당신은 질문의 복잡도를 판별하는 분류기입니다.

질문을 다음 두 가지 카테고리로 분류하세요:

1. **simple (간단한 질문)**:
   - 일반적인 학습 방법, 공부 조언, 학습 전략에 대한 질문
   - 특정 대학명, 연도, 일정이 포함되지 않은 일반적인 질문
   - 예: "수학 공부는 어떻게 해야 할까요?", "오답노트 정리법", "영어 단어 암기법"

2. **complex (복잡한 질문)**:
   - 특정 대학명이 포함된 질문 (예: 중앙대, 연세대, 고려대 등)
   - 특정 연도나 일정을 물어보는 질문 (예: 2025학년도, 시험 일정)
   - 구체적인 전형/모집요강/시험 과목 등을 물어보는 질문
   - 검색이나 데이터베이스 조회가 필요한 질문

판단 근거를 reason 필드에 간단히 적어주세요."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "다음 질문을 분류하세요:\n\n{question}")
        ])
        
        # 판별 실행
        chain = prompt | structured_llm
        result = chain.invoke({"question": question})
        
        if verbose:
            print(f"\n질문 복잡도 판별:")
            print(f"   - 질문: {question}")
            print(f"   - 판단: {result.complexity}")
            print(f"   - 이유: {result.reason}\n")
        
        return result.complexity == "simple"
        
    except Exception as e:
        if verbose:
            print(f"질문 복잡도 판별 오류 (기본값: complex): {str(e)[:100]}")
        # 오류 시 안전하게 복잡한 질문으로 처리 (기존 LangGraph 사용)
        return False


# ======================================
# 🎯 파인튜닝 모델 답변 재가공 함수
# ======================================
def refine_finetuned_answer(question: str, raw_answer: str, verbose: bool = True) -> str:
    """
    파인튜닝 모델의 원시 답변을 LLM으로 재가공하여 더 완성도 높은 답변으로 만듭니다.
    
    Parameters:
    -----------
    question : str
        원본 질문
    raw_answer : str
        파인튜닝 모델이 생성한 원시 답변
    verbose : bool
        상세 로그 출력 여부
    
    Returns:
    --------
    str
        재가공된 답변
    """
    from langchain_core.prompts import ChatPromptTemplate
    
    try:
        # 재가공 프롬프트
        system_prompt = """당신은 편입 상담 전문가입니다. 파인튜닝 모델이 생성한 답변을 받아서 간결하고 도움이 되는 답변으로 재가공해주세요.

다음 기준으로 답변을 개선하세요:

**개선 방향:**
1. **간결성**: 핵심 내용만 1-3줄로 간단명료하게 정리
2. **구체성**: 모호한 표현을 구체적이고 실행 가능한 조언으로 변경
3. **실용성**: 학생이 실제로 따라할 수 있는 구체적인 방법 제시
4. **자연스러운 표현**: 부자연스러운 부분을 자연스럽고 읽기 쉽게 수정

**주의사항:**
- 답변은 반드시 1-3줄 이내로 작성
- 핵심 내용만 간단명료하게 전달
- 편입 상담에 적합한 전문적인 톤 유지
- 구체적이고 실행 가능한 조언 제공
- 특수문자나 불필요한 형식 제거

원본 질문과 파인튜닝 모델의 답변을 바탕으로 간결한 답변을 작성해주세요."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "다음 질문과 파인튜닝 모델의 답변을 바탕으로 개선된 답변을 작성해주세요:\n\n[질문]\n{question}\n\n[파인튜닝 모델 답변]\n{raw_answer}\n\n[개선된 답변]")
        ])
        
        # 재가공 실행
        chain = prompt | llm
        refined_answer = chain.invoke({"question": question, "raw_answer": raw_answer})
        
        # 특수문자 및 불필요한 형식 제거
        if hasattr(refined_answer, 'content'):
            refined_answer = refined_answer.content
        elif isinstance(refined_answer, str):
            # content= 같은 특수문자 제거
            refined_answer = refined_answer.replace('content=', '').strip()
            # 따옴표 제거
            if refined_answer.startswith("'") and refined_answer.endswith("'"):
                refined_answer = refined_answer[1:-1]
            elif refined_answer.startswith('"') and refined_answer.endswith('"'):
                refined_answer = refined_answer[1:-1]
        
        if verbose:
            print(f"\n파인튜닝 답변 재가공:")
            print(f"   - 원본 질문: {question}")
            print(f"   - 원시 답변: {raw_answer[:100]}{'...' if len(raw_answer) > 100 else ''}")
            print(f"   - 재가공 완료: {len(refined_answer)}자\n")
        
        return refined_answer
        
    except Exception as e:
        if verbose:
            print(f"답변 재가공 오류 (원본 답변 사용): {str(e)[:100]}")
        # 오류 시 원본 답변 반환
        return raw_answer


# ======================================
# 🎯 파인튜닝 모델 답변 품질 평가 함수
# ======================================
def evaluate_answer_quality(question: str, answer: str, verbose: bool = True) -> bool:
    """
    파인튜닝 모델의 답변이 질문에 적절히 답변했는지 평가합니다.
    
    Parameters:
    -----------
    question : str
        원본 질문
    answer : str
        파인튜닝 모델이 생성한 답변
    verbose : bool
        상세 로그 출력 여부
    
    Returns:
    --------
    bool
        True: 답변이 충분히 좋음 (품질 기준 통과)
        False: 답변이 부족함 (LangGraph로 재시도 필요)
    """
    from langchain_core.prompts import ChatPromptTemplate
    
    try:
        # 구조화된 출력을 위한 LLM 설정
        structured_llm = llm.with_structured_output(AnswerQuality)
        
        # 평가 프롬프트
        system_prompt = """당신은 편입 상담 답변의 품질을 평가하는 전문가입니다.

다음 기준으로 답변을 평가하세요:

**좋은 답변 (good) 기준:**
- 질문에 직접적이고 구체적으로 답변함
- 실용적이고 실행 가능한 조언을 제공함
- 편입 상담에 적합한 전문적인 내용임
- 답변이 충분히 상세하고 도움이 됨
- 오류나 부정확한 정보가 없음

**부족한 답변 (poor) 기준:**
- 질문에 대한 답변이 모호하거나 불완전함
- "모르겠습니다", "확인해보세요" 등으로 끝남
- 너무 짧거나 일반적인 내용만 포함
- 질문과 관련 없는 내용임
- 오류나 부정확한 정보가 포함됨

점수 기준:
- 8-10점: 매우 좋은 답변 (good)
- 6-7점: 보통 답변 (good)
- 4-5점: 부족한 답변 (poor)
- 1-3점: 매우 부족한 답변 (poor)

quality 필드에 "good" 또는 "poor"를, score 필드에 1-10 점수를, reason 필드에 평가 근거를 적어주세요."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "다음 질문과 답변을 평가하세요:\n\n[질문]\n{question}\n\n[답변]\n{answer}")
        ])
        
        # 평가 실행
        chain = prompt | structured_llm
        result = chain.invoke({"question": question, "answer": answer})
        
        if verbose:
            print(f"\n답변 품질 평가:")
            print(f"   - 질문: {question}")
            print(f"   - 답변: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"   - 품질: {result.quality}")
            print(f"   - 점수: {result.score}/10")
            print(f"   - 이유: {result.reason}\n")
        
        # 6점 이상이면 좋은 답변으로 판단
        return result.score >= 6
        
    except Exception as e:
        if verbose:
            print(f"답변 품질 평가 오류 (기본값: poor): {str(e)[:100]}")
        # 오류 시 안전하게 부족한 답변으로 처리 (LangGraph 사용)
        return False


# ======================================
# 🎓 파인튜닝 모델 API 호출 함수
# ======================================
def call_finetuned_model(
    question: str,
    max_tokens: int = 100,
    temperature: float = 0.3,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    timeout: int = 120,
    max_retries: int = 3
) -> str:
    """
    CSmart-FAQ 파인튜닝 모델 API를 호출하여 답변을 생성합니다.
    
    Parameters:
    -----------
    question : str
        질문 내용
    max_tokens : int
        생성할 답변의 최대 토큰 수 (기본값: 100)
    temperature : float
        답변의 다양성 (0.1~1.0, 기본값: 0.3)
    top_k : int
        Top-K 샘플링 (기본값: 50)
    top_p : float
        Top-P (nucleus) 샘플링 (기본값: 0.95)
    repetition_penalty : float
        반복 페널티 (기본값: 1.2)
    timeout : int
        타임아웃 (초, 기본값: 120)
    max_retries : int
        최대 재시도 횟수 (기본값: 3)
    
    Returns:
    --------
    str
        생성된 답변 또는 오류 메시지
    """
    url = "https://csmart-ai-faq-finetuning.hf.space/predict"
    
    payload = {
        "question": question,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }
    
    for attempt in range(max_retries):
        try:
            print(f"파인튜닝 모델 호출 중... (시도 {attempt + 1}/{max_retries})")
            
            response = requests.post(url, json=payload, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "답변을 생성할 수 없습니다.")
                print("파인튜닝 모델 답변 생성 완료")
                return answer
                
            elif response.status_code == 400:
                error_msg = "잘못된 요청입니다. 파라미터를 확인해주세요."
                print(f"{error_msg}")
                return f"오류: {error_msg}"
                
            elif response.status_code == 500:
                print(f"서버 오류 발생 (시도 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue  # 재시도
                else:
                    return "오류: 서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            else:
                print(f"예상치 못한 오류: {response.status_code}")
                return f"오류: 예상치 못한 오류 (상태 코드: {response.status_code})"
                
        except requests.exceptions.Timeout:
            print(f"요청 시간 초과 (시도 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                continue  # 재시도
            else:
                return "오류: 요청 시간이 초과되었습니다."
                
        except requests.exceptions.RequestException as e:
            print(f"네트워크 오류: {str(e)[:100]}")
            if attempt < max_retries - 1:
                continue  # 재시도
            else:
                return f"오류: 네트워크 오류 ({str(e)[:50]})"
    
    return "오류: 최대 재시도 횟수를 초과했습니다."


# ======================================
# 🎯 메인 API 함수 (🆕 라우팅 로직 포함)
# ======================================
def get_answer(
    question: str,
    student_profile: Optional[Dict[str, str]] = None,
    recent_dialogues: Optional[List[Dict[str, str]]] = None,
    verbose: bool = True,
    force_mode: Optional[Literal["simple", "complex"]] = None
) -> Dict:
    """
    편입 상담 질문에 대한 답변을 생성합니다.
    
    질문의 복잡도에 따라 자동으로 라우팅됩니다:
    - 간단한 질문 (일반적인 학습 조언) → 파인튜닝 모델 사용 → LLM 재가공 → 답변 품질 평가 → 기준 미달 시 LangGraph 재시도
    - 복잡한 질문 (특정 대학/일정/전형 정보) → LangGraph 에이전트 사용
    
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
    
    force_mode : Literal["simple", "complex"], optional
        수동으로 특정 모드를 선택하도록 지정 (기본값: None, 자동 판별)
        - "simple": 파인튜닝 모델 수동 선택
        - "complex": LangGraph 에이전트 수동 선택
    
    Returns:
    --------
    dict
        {
            "question": str,           # 원본 질문
            "final_answer": str,       # 최종 답변
            "model_used": str,         # 사용된 모델 ("finetuned_refined", "langgraph", "langgraph_fallback")
            "context": str,            # 생성된 컨텍스트 (LangGraph 사용 시)
            "datasources": list,       # 사용된 데이터 소스 (LangGraph 사용 시)
            "success": bool,           # 성공 여부
            "error": str or None       # 오류 메시지 (있는 경우)
        }
    
    Examples:
    ---------
    >>> from api import get_answer
    >>> 
    >>> # 간단한 질문 (자동으로 파인튜닝 모델 사용)
    >>> result = get_answer("수학 공부는 어떻게 해야 할까요?")
    >>> print(f"사용된 모델: {result['model_used']}")  # "finetuned_refined"
    >>> print(result["final_answer"])
    >>> 
    >>> # 복잡한 질문 (자동으로 LangGraph 사용)
    >>> result = get_answer("중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
    >>> print(f"사용된 모델: {result['model_used']}")  # "langgraph"
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
    >>> 
    >>> # 수동으로 특정 모드 선택
    >>> result = get_answer("수학 공부법", force_mode="complex")  # 수동으로 LangGraph 선택
    """
    
    # 기본값 설정
    if student_profile is None:
        student_profile = {
            "target_university": "미지정",
            "track": "계열 미지정"
        }
    
    if recent_dialogues is None:
        recent_dialogues = []
    
    try:
        # 로그 출력 제어
        if not verbose:
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
        
        # ==========================================
        # 🔀 1단계: 질문 복잡도 판별 및 라우팅
        # ==========================================
        if force_mode:
            # 수동 선택 모드가 지정된 경우
            use_simple_model = (force_mode == "simple")
            if verbose or force_mode:
                print(f"\n수동 선택 모드: {force_mode}")
        else:
            # 자동 판별
            use_simple_model = is_simple_question(question, verbose=verbose)
        
        # ==========================================
        # 🎓 2단계: 간단한 질문 → 파인튜닝 모델 사용
        # ==========================================
        if use_simple_model:
            print("\n" + "="*60)
            print("라우팅 결정: 파인튜닝 모델 사용 (간단한 질문)")
            print("="*60)
            
            # 파인튜닝 모델 호출
            answer = call_finetuned_model(
                question=question,
                max_tokens=100,
                temperature=0.3,
                timeout=120,
                max_retries=3
            )
            
            # 파인튜닝 모델 답변 재가공 및 품질 평가
            if not answer.startswith("오류:"):
                # 1단계: 파인튜닝 답변을 LLM으로 재가공
                refined_answer = refine_finetuned_answer(question, answer, verbose=verbose)
                
                # 2단계: 재가공된 답변의 품질 평가
                is_good_answer = evaluate_answer_quality(question, refined_answer, verbose=verbose)
                
                if is_good_answer:
                    # 품질이 좋으면 재가공된 답변 사용
                    print("파인튜닝 모델 답변 재가공 및 품질 통과 - 최종 답변으로 사용")
                    
                    if not verbose:
                        sys.stdout = old_stdout
                    
                    return {
                        "question": question,
                        "final_answer": refined_answer,
                        "model_used": "finetuned_refined",
                        "context": "",
                        "datasources": ["finetuned_model", "llm_refinement"],
                        "success": True,
                        "error": None
                    }
                else:
                    # 품질이 부족하면 LangGraph로 재시도
                    print("파인튜닝 모델 답변 품질 미달 - LangGraph로 재시도")
                    print("\n" + "="*60)
                    print("재라우팅: LangGraph 에이전트 사용 (답변 품질 미달)")
                    print("="*60)
                    
                    # 입력 데이터 구성
                    inputs = {
                        "question": question,
                        "student_profile": student_profile,
                        "recent_dialogues": recent_dialogues
                    }
                    
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
                        "model_used": "langgraph_fallback",
                        "context": result.get("context", ""),
                        "datasources": result.get("datasources", []),
                        "success": True,
                        "error": None
                    }
            else:
                # 파인튜닝 모델 오류 시 LangGraph로 재시도
                print("파인튜닝 모델 오류 - LangGraph로 재시도")
                print("\n" + "="*60)
                print("재라우팅: LangGraph 에이전트 사용 (파인튜닝 모델 오류)")
                print("="*60)
                
                # 입력 데이터 구성
                inputs = {
                    "question": question,
                    "student_profile": student_profile,
                    "recent_dialogues": recent_dialogues
                }
                
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
                    "model_used": "langgraph_fallback",
                    "context": result.get("context", ""),
                    "datasources": result.get("datasources", []),
                    "success": True,
                    "error": None
                }
        
        # ==========================================
        # 🤖 3단계: 복잡한 질문 → LangGraph 에이전트 사용
        # ==========================================
        else:
            print("\n" + "="*60)
            print("라우팅 결정: LangGraph 에이전트 사용 (복잡한 질문)")
            print("="*60)
            
            # 입력 데이터 구성
            inputs = {
                "question": question,
                "student_profile": student_profile,
                "recent_dialogues": recent_dialogues
            }
            
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
                "model_used": "langgraph",
                "context": result.get("context", ""),
                "datasources": result.get("datasources", []),
                "success": True,
                "error": None
            }
        
    except Exception as e:
        if not verbose:
            sys.stdout = old_stdout
        
        error_msg = str(e)
        print(f"오류 발생: {error_msg[:200]}")
        
        return {
            "question": question,
            "final_answer": "오류로 인해 답변을 생성하지 못했습니다.",
            "model_used": "error",
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
    print("CSmart API 테스트 (라우팅 기능 포함)")
    print("=" * 80)
    
    # 테스트 1: 간단한 질문 (파인튜닝 모델 사용)
    print("\n[테스트 1] 간단한 질문 - 파인튜닝 모델")
    print("-" * 80)
    result1 = get_answer(
        question="수학 공부는 어떻게 해야 할까요?",
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("테스트 1 결과:")
    print("=" * 80)
    print(f"질문: {result1['question']}")
    print(f"사용된 모델: {result1['model_used']}")
    print(f"성공 여부: {result1['success']}")
    print(f"\n답변:\n{result1['final_answer']}")
    
    # 테스트 2: 복잡한 질문 (LangGraph 사용)
    print("\n\n[테스트 2] 복잡한 질문 - LangGraph 에이전트")
    print("-" * 80)
    result2 = get_answer(
        question="중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?",
        student_profile={"target_university": "중앙대학교", "track": "이과"},
        recent_dialogues=[
            {"role": "student", "message": "편입 준비를 시작하려고 합니다."},
            {"role": "teacher", "message": "좋습니다. 어느 대학을 목표로 하시나요?"},
            {"role": "student", "message": "중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?"}
        ],
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("테스트 2 결과:")
    print("=" * 80)
    print(f"질문: {result2['question']}")
    print(f"사용된 모델: {result2['model_used']}")
    print(f"성공 여부: {result2['success']}")
    print(f"사용된 데이터 소스: {result2['datasources']}")
    print(f"\n답변:\n{result2['final_answer']}")
    
    # 테스트 3: 수동 선택 모드 테스트
    print("\n\n[테스트 3] 수동 선택 모드 - 간단한 질문을 LangGraph로")
    print("-" * 80)
    result3 = get_answer(
        question="영어 단어 암기는 어떻게 해야 할까요?",
        force_mode="complex",  # 수동으로 LangGraph 선택
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("테스트 3 결과:")
    print("=" * 80)
    print(f"질문: {result3['question']}")
    print(f"사용된 모델: {result3['model_used']}")
    print(f"성공 여부: {result3['success']}")
    
    print("\n" + "=" * 80)
    print("모든 API 테스트 완료")
    print("=" * 80)
    print("\n[요약]")
    print(f"테스트 1 (간단한 질문): {result1['model_used']} 사용")
    print(f"테스트 2 (복잡한 질문): {result2['model_used']} 사용")
    print(f"테스트 3 (수동 선택 모드): {result3['model_used']} 사용")
    print("=" * 80)
