#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이모지 제거 스크립트
"""

import os
import re

def remove_emojis_from_file(file_path):
    """파일에서 이모지를 제거합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 이모지 패턴 정의
        emoji_pattern = r'[🚀✅❌⚠️📌🔍🎓🤖📚🌐📝📋🔒🆕]'
        
        # 이모지 제거
        cleaned_content = re.sub(emoji_pattern, '', content)
        
        # 변경사항이 있으면 파일 저장
        if cleaned_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"이모지 제거 완료: {file_path}")
            return True
        else:
            print(f"변경사항 없음: {file_path}")
            return False
            
    except Exception as e:
        print(f"오류 발생 {file_path}: {e}")
        return False

def main():
    """메인 함수"""
    # 처리할 파일 목록
    files_to_process = [
        'step3_db_and_search.py',
        'step5_guideline_agent.py',
        'step6_web_agent.py',
        'step7_integrated_agent.py',
        'step8_test.py',
        'test_routing.py'
    ]
    
    print("이모지 제거 시작...")
    
    for file_path in files_to_process:
        if os.path.exists(file_path):
            remove_emojis_from_file(file_path)
        else:
            print(f"파일 없음: {file_path}")
    
    print("이모지 제거 완료!")

if __name__ == "__main__":
    main()
