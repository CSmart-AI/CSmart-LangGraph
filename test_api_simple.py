#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 API 테스트 스크립트
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from api import get_answer
    
    print("=" * 80)
    print("CSmart API 테스트 (답변 품질 평가 기능 포함)")
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
    
    print("\n" + "=" * 80)
    print("모든 API 테스트 완료")
    print("=" * 80)
    print("\n[요약]")
    print(f"테스트 1 (간단한 질문): {result1['model_used']} 사용")
    print(f"테스트 2 (복잡한 질문): {result2['model_used']} 사용")
    print("=" * 80)
    
except ImportError as e:
    print(f"Import 오류: {e}")
    print("필요한 모듈이 설치되지 않았을 수 있습니다.")
except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()
