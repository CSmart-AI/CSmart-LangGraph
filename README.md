# CSmart 편입 상담 AI 시스템

## 🏗️ 프로젝트 구조

```
CSmart/
├── 📓 CSmart_LangGraph.ipynb       # 원본 Jupyter Notebook (ipynb)
├── 🎯 api.py                       # 메인 API (ipynb를 모듈화)
├── 📝 example_usage.py             # 사용 예제 모음
│
├── step1_env.py                    # 환경 변수 로드 (Cell 2)
├── step2_states.py                 # 상태 정의 및 prepare_context (Cell 4-5)
├── step3_db_and_search.py          # DB 및 검색 도구 (Cell 9, 하이브리드 검색)
├── step4_llm.py                    # LLM 설정 (Cell 14, Gemini)
├── step5_guideline_agent.py        # GuidelineDB RAG 에이전트 (Cell 19)
├── step6_web_agent.py              # 웹 검색 RAG 에이전트 (Cell 22)
├── step7_integrated_agent.py       # 통합 에이전트 (Cell 25-27, 라우팅)
├── step8_test.py                   # 원본 모듈별 테스트 (Cell 28)
│
├── GuidelineDB.csv                 # 가이드라인 데이터
├── chroma_guideline/               # 벡터 DB 저장소
├── requirements.txt                # 필요 패키지
├── .env                            # 환경 변수
└── README.md                       # 이 파일
```



## 💡 중요: ipynb vs api.py

이 프로젝트는 **두 가지 방식**으로 사용가능:

1. **`CSmart_LangGraph.ipynb`** (Jupyter Notebook)
   - 원본 개발 파일
   - 각 셀을 단계별로 실행하며 확인 가능

2. **`api.py`** (Python 스크립트)
   - ipynb를 모듈화한 버전
   

**⚠️ 두 파일은 100% 동일한 로직을 사용합니다!**
- ipynb의 모든 코드가 step1-8.py로 분리됨
- api.py는 이를 하나로 통합한 인터페이스
- **아무거나 사용해도 결과는 같습니다**

---

## 🚀 빠른 시작

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 다음 내용을 추가:
```
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Python 스크립트 사용

```python
from api import get_answer

# 기본 사용 (로그 출력됨)
result = get_answer("중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
print(result["final_answer"])

# 로그 숨기기
result = get_answer("질문내용", verbose=False)
print(result["final_answer"])

```
### api.py에서 get_answer() 함수 하나로 모든 기능을 사용할 수 있습니다.

사용법:
    'from api import get_answer'
    
    result = get_answer("중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
    print(result["final_answer"])


**Parameters:**
- `question` (str): 학생의 질문
- `student_profile` (dict, optional): 학생 프로필
  - `target_university`: 목표 대학명
  - `track`: 계열 (이과/문과)
- `recent_dialogues` (list, optional): 최근 대화 내역
  - 각 항목: `{"role": "student"|"teacher", "message": "..."}`
- `verbose` (bool, optional): 로그 출력 여부 (기본값: True)

**Returns:**
```python
{
    "question": "원본 질문",
    "final_answer": "최종 답변",
    "context": "생성된 컨텍스트",
    "datasources": ["사용된", "데이터", "소스"],
    "success": True,
    "error": None
}
```

---



## 🔄 실제 동작 과정

`python api.py` 실행 시:

```
1. 🚀 API 초기화
   └─ step1~7 모든 모듈 자동 로드
   └─ Chroma DB, LLM, 에이전트 자동 컴파일

2. 📋 컨텍스트 준비 (step2)
   └─ 학생 프로필 + 대화 내역 → context 생성

3. 🔍 질문 분석 (step7)
   └─ LLM이 적절한 도구 선택
   
4. 📚 GuidelineDB 검색 (step5, 선택 시)
   ├─ [1단계] 하이브리드 검색으로 문서 검색
   ├─ [2단계] LLM이 정보 추출 및 평가
   ├─ [판단] 정보 충분 여부 확인
   ├─ [3단계] 부족 시 쿼리 재작성 (최대 2회)
   └─ [4단계] 최종 답변 생성

5. 🌐 웹 검색 (step6, 선택 시)
   ├─ Tavily API로 문서 검색
   ├─ 정보 추출 및 평가
   ├─ 부족 시 쿼리 재작성
   └─ 최종 답변 생성

6. 📝 최종 답변 생성 (step7)
   └─ 수집된 정보를 종합하여 학생에게 제공
```
