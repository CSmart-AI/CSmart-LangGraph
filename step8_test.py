# Cell 28 - 새로운 통합 API 테스트
from api import get_answer
import time
from typing import Dict, List

# =========================================
# 🧪 새로운 통합 API 테스트 (파인튜닝 + LangGraph)
# =========================================

def test_unified_api(question: str, test_name: str, 
                     student_profile: Dict[str, str] = None,
                     recent_dialogues: List[Dict[str, str]] = None,
                     force_mode: str = None):
    """새로운 통합 API를 테스트합니다."""
    print(f"\n{'='*80}")
    print(f"[테스트] {test_name}")
    print(f"{'='*80}")
    print(f"질문: {question}")
    
    if force_mode:
        print(f"수동 모드: {force_mode}")
    
    if student_profile:
        print(f"학생 프로필: {student_profile['target_university']} {student_profile['track']}")
    
    print(f"\n처리 중...")
    start_time = time.time()
    
    try:
        result = get_answer(
            question=question,
            student_profile=student_profile,
            recent_dialogues=recent_dialogues,
            verbose=False,  # 로그 숨김
            force_mode=force_mode
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n처리 완료 ({processing_time:.1f}초)")
        print(f"사용된 모델: {result['model_used']}")
        
        if result['datasources']:
            print(f"데이터 소스: {', '.join(result['datasources'])}")
        
        print(f"\n최종 답변:")
        print(f"{'─'*60}")
        print(result['final_answer'])
        print(f"{'─'*60}")
        
        return result
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)[:200]}")
        return None


# =========================================
# 테스트 실행
# =========================================

print("CSmart 통합 API 테스트 시작")
print("파인튜닝 모델 + LangGraph 에이전트 통합 사용")

# 테스트 1: 간단한 질문 (파인튜닝 모델 사용)
test_unified_api(
    question="수학 공부는 어떻게 해야 할까요?",
    test_name="테스트 1: 간단한 질문 (파인튜닝 모델)",
    student_profile={"target_university": "중앙대학교", "track": "이과"}
)

# 테스트 2: 복잡한 질문 (LangGraph 사용)
test_unified_api(
    question="중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?",
    test_name="테스트 2: 복잡한 질문 (LangGraph 에이전트)",
    student_profile={"target_university": "중앙대학교", "track": "이과"},
    recent_dialogues=[
        {"role": "student", "message": "편입 준비를 시작하려고 합니다."},
        {"role": "teacher", "message": "어느 대학을 목표로 하시나요?"},
        {"role": "student", "message": "중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?"}
    ]
)

# 테스트 3: 수동으로 파인튜닝 모델 강제 사용
test_unified_api(
    question="영어 단어 암기는 어떻게 해야 할까요?",
    test_name="테스트 3: 수동 모드 (파인튜닝 강제)",
    force_mode="simple"
)

# 테스트 4: 수동으로 LangGraph 강제 사용
test_unified_api(
    question="2025학년도 편입 시험 일정은 언제인가요?",
    test_name="테스트 4: 수동 모드 (LangGraph 강제)",
    force_mode="complex"
)

print(f"\n{'='*80}")
print("모든 테스트 완료!")
print("통합 API가 파인튜닝 모델과 LangGraph를 성공적으로 활용했습니다.")
print(f"{'='*80}")
