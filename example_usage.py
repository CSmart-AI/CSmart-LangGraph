"""
CSmart API 사용 예제 모음
"""

from api import get_answer

print("=" * 80)
print("CSmart API 사용 예제")
print("=" * 80)


# ============================================================
# 예제 1: 가장 간단한 사용법 (로그 출력)
# ============================================================
print("\n\n[예제 1] 가장 간단한 사용법 (로그 출력)")
print("-" * 80)

result = get_answer("중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
print(f"\n답변: {result['final_answer']}")


# ============================================================
# 예제 2: 학생 프로필 포함
# ============================================================
print("\n\n[예제 2] 학생 프로필 포함")
print("-" * 80)

result = get_answer(
    question="편입 준비는 어떻게 시작하나요?",
    student_profile={
        "target_university": "중앙대학교",
        "track": "이과"
    }
)

print(f"\n질문: {result['question']}")
print(f"답변: {result['final_answer']}")
print(f"사용된 데이터 소스: {result['datasources']}")


# ============================================================
# 예제 3: 대화 내역 포함 (컨텍스트 활용)
# ============================================================
print("\n\n[예제 3] 대화 내역 포함")
print("-" * 80)

result = get_answer(
    question="그럼 수학은 어떻게 공부해야 하나요?",
    student_profile={
        "target_university": "중앙대학교",
        "track": "이과"
    },
    recent_dialogues=[
        {"role": "student", "message": "편입 준비를 시작하려고 합니다."},
        {"role": "teacher", "message": "좋습니다. 어느 대학을 목표로 하시나요?"},
        {"role": "student", "message": "중앙대학교 이과입니다."},
        {"role": "teacher", "message": "중앙대 이과는 수학 시험만 봅니다."},
        {"role": "student", "message": "그럼 수학은 어떻게 공부해야 하나요?"}
    ]
)

print(f"\n컨텍스트:\n{result['context']}\n")
print(f"답변: {result['final_answer']}")


# ============================================================
# 예제 4: 로그 숨기기 (verbose=False)
# ============================================================
print("\n\n[예제 4] 로그 숨기기")
print("-" * 80)

result = get_answer(
    question="편입영어는 어떻게 공부하나요?",
    verbose=False  # 로그 출력 안 함
)

print(f"답변: {result['final_answer']}")


# ============================================================
# 예제 5: 여러 질문 연속 처리
# ============================================================
print("\n\n[예제 5] 여러 질문 연속 처리")
print("-" * 80)

questions = [
    "중앙대학교 이과 편입 과목은?",
    "편입영어는 어떻게 공부하나요?",
    "수학 공부 시간 배분은?"
]

for i, q in enumerate(questions, 1):
    print(f"\n질문 {i}: {q}")
    result = get_answer(q, verbose=False)  # 로그 숨김
    print(f"답변: {result['final_answer'][:100]}...")


# ============================================================
# 예제 6: 오류 처리
# ============================================================
print("\n\n[예제 6] 오류 처리")
print("-" * 80)

result = get_answer(
    question="테스트 질문",
    student_profile={"target_university": "테스트대학"},
    verbose=False
)

if result['success']:
    print(f"✅ 성공: {result['final_answer'][:100]}...")
else:
    print(f"❌ 실패: {result['error']}")


print("\n\n" + "=" * 80)
print("✅ 모든 예제 완료")
print("=" * 80)