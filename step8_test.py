# Cell 28
from typing import Dict, List
from step7_integrated_agent import integrated_agent

# =========================================
# 🧪 통합 에이전트 테스트 실행 (✅ 컨텍스트 포함)
# =========================================

def test_integrated_agent(question: str, test_name: str, 
                          student_profile: Dict[str, str] = None,
                          recent_dialogues: List[Dict[str, str]] = None):
    """안전하게 통합 에이전트를 테스트 (컨텍스트 포함)"""
    print("\n" + "="*60)
    print(f"{test_name}")
    print("="*60)
    
    try:
        # 기본 학생 프로필 설정
        if student_profile is None:
            student_profile = {
                "target_university": "중앙대학교",
                "track": "이과"  # ✅ track 필드 사용 (Cell 6 기준)
            }
        
        # 기본 대화 내역 설정
        if recent_dialogues is None:
            recent_dialogues = [
                {"role": "student", "message": "아직 편입 공부를 시작하진 않았습니다."},
                {"role": "teacher", "message": "언제부터 공부 시작할 계획인가요?"},
                {"role": "student", "message": "다음 주부터 시작할 생각입니다."},
                {"role": "teacher", "message": "네 알겠습니다."},
                {"role": "student", "message": question}
            ]
        
        inputs = {
            "question": question,
            "student_profile": student_profile,
            "recent_dialogues": recent_dialogues
        }
        
        result = integrated_agent.invoke(
            inputs,
            config={
                "recursion_limit": 25,  # 🔒 전체 재귀 제한
                "timeout": 120  # 🔒 2분 타임아웃
            }
        )
        
        print("\n✅ 최종 답변:")
        print(result.get("final_answer", "답변을 생성하지 못했습니다."))
        return result
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)[:200]}")
        return None


# 테스트 1: GuidelineDB 검색이 필요한 질문
test_integrated_agent(
    question="중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?",
    test_name="테스트 1: GuidelineDB 검색",
    student_profile={"target_university": "중앙대학교", "track": "이과"}  # ✅ track 필드 사용
)

# 테스트 2: 웹 검색이 필요한 질문  
test_integrated_agent(
    question="2025학년도 중앙대학교 편입 시험 일정은 언제였나요?",
    test_name="테스트 2: 웹 검색",
    student_profile={"target_university": "중앙대학교", "track": "이과"},
    recent_dialogues=[
        {"role": "student", "message": "편입 시험 일정이 궁금합니다."},
        {"role": "teacher", "message": "어느 대학을 알아보고 계신가요?"},
        {"role": "student", "message": "2025학년도 중앙대학교 편입 시험 일정은 언제였나요?"}
    ]
)

# 테스트 3: 두 가지 모두 필요할 수 있는 질문
test_integrated_agent(
    question="중앙대학교 이과 편입 시험 과목과 일정을 알려주세요",
    test_name="테스트 3: 복합 질문",
    student_profile={"target_university": "중앙대학교", "track": "이과"},
    recent_dialogues=[
        {"role": "student", "message": "편입 준비를 막 시작했습니다."},
        {"role": "teacher", "message": "어떤 정보가 필요하신가요?"},
        {"role": "student", "message": "중앙대학교 이과 편입 시험 과목과 일정을 알려주세요"}
    ]
)

print("\n" + "="*60)
print("✅ 모든 테스트 완료 (컨텍스트 활용)")
print("="*60)
