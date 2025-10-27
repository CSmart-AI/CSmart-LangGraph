"""
🧪 라우팅 기능 테스트 스크립트

이 스크립트는 새로운 질문 라우팅 기능을 테스트합니다:
- 간단한 질문 → 파인튜닝 모델 사용
- 복잡한 질문 → LangGraph 에이전트 사용
"""

from api import get_answer


def test_simple_question():
    """간단한 질문 테스트 (파인튜닝 모델 사용 예상)"""
    print("\n" + "="*80)
    print("테스트 1: 간단한 질문")
    print("="*80)
    
    question = "오답노트는 어떻게 작성해야 하나요?"
    print(f"질문: {question}")
    
    result = get_answer(question, verbose=True)
    
    print("\n[결과]")
    print(f"사용된 모델: {result['model_used']}")
    print(f"성공 여부: {result['success']}")
    print(f"\n답변:\n{result['final_answer'][:200]}...")
    
    return result


def test_complex_question():
    """복잡한 질문 테스트 (LangGraph 사용 예상)"""
    print("\n" + "="*80)
    print("테스트 2: 복잡한 질문")
    print("="*80)
    
    question = "중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?"
    print(f"질문: {question}")
    
    result = get_answer(
        question=question,
        student_profile={"target_university": "중앙대학교", "track": "이과"},
        verbose=True
    )
    
    print("\n[결과]")
    print(f"사용된 모델: {result['model_used']}")
    print(f"성공 여부: {result['success']}")
    print(f"데이터 소스: {result['datasources']}")
    print(f"\n답변:\n{result['final_answer'][:200]}...")
    
    return result


def test_force_mode():
    """강제 모드 테스트 (간단한 질문을 LangGraph로)"""
    print("\n" + "="*80)
    print("테스트 3: 강제 모드")
    print("="*80)
    
    question = "영어 단어 암기는 어떻게 해야 할까요?"
    print(f"질문: {question}")
    print("강제 모드: complex (LangGraph 강제 사용)")
    
    result = get_answer(
        question=question,
        force_mode="complex",  # 강제로 LangGraph 사용
        verbose=True
    )
    
    print("\n[결과]")
    print(f"사용된 모델: {result['model_used']}")
    print(f"성공 여부: {result['success']}")
    
    return result


def test_additional_questions():
    """추가 테스트 질문들"""
    print("\n" + "="*80)
    print("테스트 4: 다양한 질문 테스트")
    print("="*80)
    
    test_questions = [
        ("오답노트는 어떻게 정리할까요?", "simple 예상"),
        ("2025학년도 중앙대 편입 일정은?", "complex 예상"),
        ("공부 집중력을 높이는 방법은?", "simple 예상"),
        ("연세대학교 편입 모집요강을 알려주세요", "complex 예상"),
    ]
    
    results = []
    for question, expected in test_questions:
        print(f"\n질문: {question} ({expected})")
        result = get_answer(question, verbose=False)
        print(f"→ 사용된 모델: {result['model_used']}")
        results.append((question, result['model_used']))
    
    return results


if __name__ == "__main__":
    print("\n" + "🎯"*40)
    print("CSmart 라우팅 기능 테스트 시작")
    print("🎯"*40)
    
    # 테스트 1: 간단한 질문
    result1 = test_simple_question()
    
    # 테스트 2: 복잡한 질문
    result2 = test_complex_question()
    
    # 테스트 3: 강제 모드
    result3 = test_force_mode()
    
    # 테스트 4: 다양한 질문
    result4 = test_additional_questions()
    
    # 최종 요약
    print("\n" + "="*80)
    print(" 테스트 요약")
    print("="*80)
    print(f"테스트 1 (간단한 질문): {result1['model_used']}")
    print(f"테스트 2 (복잡한 질문): {result2['model_used']}")
    print(f"테스트 3 (강제 모드): {result3['model_used']}")
    print("\n테스트 4 (다양한 질문):")
    for q, model in result4:
        print(f"  - {q[:30]}... → {model}")
    
    print("\n" + "="*80)
    print(" 모든 테스트 완료!")
    print("="*80)

