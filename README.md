# CSmart 편입 상담 AI Agent

대학 편입 준비생을 위한 상담 보조 AI Agent입니다. **파인튜닝 모델**과 **LangGraph 에이전트**를 통합한 하이브리드 AI 시스템입니다.

## 핵심 기능: 통합 AI 시스템

**하나의 API**로 두 가지 AI 모델을 함께 사용합니다

### 자동 라우팅 (Question Routing)
질문의 복잡도에 따라 **자동으로** 적절한 모델을 선택합니다

1. **간단한 질문** (일반적인 학습 조언) → **파인튜닝 모델** 사용
  - 원시 답변 생성 → **LLM 재가공** → 품질 평가 → 최종 답변
  - 빠르고 효율적, 일반적인 학습 조언에 최적화
2. **복잡한 질문** (특정 대학/일정/전형 정보) → **LangGraph 에이전트** 사용
  - GuidelineDB 검색 + 웹 검색 → 종합 분석 → 상세한 답변
  - 정확하고 상세, 검색 기반 정보 제공

### 품질 보장 시스템
- **파인튜닝 모델**: 답변 품질 평가 후 6점 미만 시 LangGraph로 자동 재라우팅
- **오류 처리**: 파인튜닝 모델 오류 시 LangGraph로 자동 폴백
- **이중 안전장치**: 어떤 상황에서도 최적의 답변 보장

**예시:**
- "수학 공부는 어떻게 해야 할까요?" → 파인튜닝 모델 (재가공 후 반환)
- "중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?" → LangGraph 에이전트
- 파인튜닝 답변 품질 미달 → 자동으로 LangGraph로 재시도

### 전체 시스템 구조
<img width="740" height="2124" alt="1  전체 시스템 구조" src="https://github.com/user-attachments/assets/54877d23-ad6c-4d64-8bdc-e21ca6d4554b" />

---

## 빠른 시작 (처음 사용자)

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. .env 파일에 API 키 설정 (GOOGLE_API_KEY, TAVILY_API_KEY)

# 3. my_question.py 파일 열어서 질문 수정 후 실행
python my_question.py
```


---

## 프로젝트 구조

```
CSmart/
├── CSmart_LangGraph.ipynb       # 원본 Jupyter Notebook (ipynb)
├── api.py                       # 통합 API (파인튜닝 + LangGraph)
├── my_question.py               # 질문 입력용 (가장 간단!)
├── example_usage.py             # 사용 예제 모음
│
├── step1_env.py                    # 환경 변수 로드 (Cell 2)
├── step2_states.py                 # 상태 정의 및 prepare_context (Cell 4-5)
├── step3_db_and_search.py          # DB 및 검색 도구 (Cell 9, 하이브리드 검색)
├── step4_llm.py                    # LLM 설정 (Cell 14, Gemini)
├── step5_guideline_agent.py        # GuidelineDB RAG 에이전트 (Cell 19)
├── step6_web_agent.py              # 웹 검색 RAG 에이전트 (Cell 22)
├── step7_integrated_agent.py       # 통합 에이전트 (Cell 25-27, 라우팅)
├── step8_test.py                   # 통합 API 테스트 (깔끔한 로그)
├── test_routing.py                 # 라우팅 기능 테스트 스크립트
│
├── GuidelineDB.csv                 # 가이드라인 데이터
├── chroma_guideline/               # 벡터 DB 저장소
├── requirements.txt                # 필요 패키지
├── .env                            # 환경 변수
├── system_architecture.md          # 시스템 아키텍처 문서
├── generate_diagrams.html          # 다이어그램 시각화
└── README.md                       # 이 파일
```



## 중요: ipynb vs api.py

이 프로젝트는 **두 가지 방식**으로 사용가능:

1. **`CSmart_LangGraph.ipynb`** (Jupyter Notebook)
   - 원본 개발 파일
   - 각 셀을 단계별로 실행하며 확인 가능

2. **`api.py`** (Python 스크립트)
   - ipynb를 모듈화한 버전
   

**두 파일은 100% 동일한 로직을 사용합니다!**
- ipynb의 모든 코드가 step1-8.py로 분리됨
- api.py는 이를 하나로 통합한 인터페이스

---

## 빠른 시작

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

### 3. 질문하는 방법

#### 방법 1: 간단하게 질문 바꾸기

**Step 1)** `my_question.py` 파일 열기

**Step 2)** 질문 내용만 수정:
```python
question = "여기에 원하는 질문 입력!"  # 이 부분만 수정
```

**Step 3)** 터미널에서 실행:
```bash
python my_question.py
```

**장점:**
- 질문만 바꾸면 됨 (가장 간단)
- 원본 파일(api.py) 안전
- 여러 질문 파일을 만들 수 있음

---

#### 방법 2: 직접 코드 작성

새로운 `.py` 파일을 만들어서 사용:

```python
from api import get_answer

# 간단한 질문 (자동으로 파인튜닝 모델 사용)
result = get_answer("수학 공부는 어떻게 해야 할까요?")
print(f"사용된 모델: {result['model_used']}")  # "finetuned_refined"
print(result["final_answer"])

# 복잡한 질문 (자동으로 LangGraph 사용)
result = get_answer("중앙대학교 이과 편입은 어떤 과목을 준비해야 하나요?")
print(f"사용된 모델: {result['model_used']}")  # "langgraph"
print(result["final_answer"])

# 로그 숨기기 (깔끔한 출력)
result = get_answer("질문내용", verbose=False)
print(result["final_answer"])

# 강제 모드 (특정 모델 지정)
result = get_answer("수학 공부법", force_mode="complex")  # LangGraph 강제 사용
result = get_answer("중앙대 편입", force_mode="simple")   # 파인튜닝 강제 사용
```

**Parameters:**
- `question` (str): 학생의 질문
- `student_profile` (dict, optional): 학생 프로필
  - `target_university`: 목표 대학명
  - `track`: 계열 (이과/문과)
- `recent_dialogues` (list, optional): 최근 대화 내역
  - 각 항목: `{"role": "student"|"teacher", "message": "..."}`
- `verbose` (bool, optional): 로그 출력 여부 (기본값: True)
- `force_mode` (Literal["simple", "complex"], optional): 모델 강제 지정
  - `"simple"`: 파인튜닝 모델 강제 사용
  - `"complex"`: LangGraph 에이전트 강제 사용
  - `None`: 자동 판별 (기본값)

**Returns:**
```python
{
    "question": "원본 질문",
    "final_answer": "최종 답변",
    "model_used": "finetuned_refined" 또는 "langgraph" 또는 "langgraph_fallback",  # 사용된 모델
    "context": "생성된 컨텍스트",
    "datasources": ["사용된", "데이터", "소스"],
    "success": True,
    "error": None
}
```

---



## 실제 동작 과정

`python api.py` 실행 시:

```
1. API 초기화
   └─ step1~7 모든 모듈 자동 로드
   └─ Chroma DB, LLM, 에이전트 자동 컴파일

2. 질문 복잡도 판별 (자동 라우팅)
   ├─ LLM이 질문 분석
   ├─ 간단한 질문? → 파인튜닝 모델 사용
   │   ├─ Hugging Face API 호출 → 원시 답변 생성
   │   ├─ LLM으로 답변 재가공 → 완성도 향상
   │   ├─ 답변 품질 평가 (1-10점)
   │   ├─ 6점 이상? → 재가공된 답변 반환
   │   └─ 6점 미만? → LangGraph로 재라우팅
   └─ 복잡한 질문? → LangGraph 에이전트 사용 (아래 3-6단계)

3. 컨텍스트 준비 (step2, LangGraph 사용 시)
   └─ 학생 프로필 + 대화 내역 → context 생성

4. 질문 분석 (step7, LangGraph 사용 시)
   └─ LLM이 적절한 도구 선택 (GuidelineDB / Web)
   
5. GuidelineDB 검색 (step5, 선택 시)
   ├─ [1단계] 하이브리드 검색으로 문서 검색
   ├─ [2단계] LLM이 정보 추출 및 평가
   ├─ [판단] 정보 충분 여부 확인
   ├─ [3단계] 부족 시 쿼리 재작성 (최대 2회)
   └─ [4단계] 최종 답변 생성

6. 웹 검색 (step6, 선택 시)
   ├─ Tavily API로 문서 검색
   ├─ 정보 추출 및 평가
   ├─ 부족 시 쿼리 재작성
   └─ 최종 답변 생성

7. 최종 답변 생성 (step7, LangGraph 사용 시)
   └─ 수집된 정보를 종합하여 학생에게 제공
```


## 테스트

### 통합 API 테스트 (추천!)
```bash
python step8_test.py
```

깔끔한 로그와 함께 통합 API를 테스트합니다:
- 간단한 질문 (파인튜닝 모델)
- 복잡한 질문 (LangGraph 에이전트)
- 수동 모드 테스트
- 처리 시간 및 사용된 모델 표시

### 라우팅 기능 테스트
```bash
python test_routing.py
```

다양한 질문으로 자동 라우팅 기능을 테스트합니다:
- 간단한 질문 (파인튜닝 모델)
- 복잡한 질문 (LangGraph)
- 강제 모드 테스트
- 다양한 질문 유형 테스트

### 시스템 아키텍처 시각화
```bash
# 브라우저에서 다이어그램 확인
start generate_diagrams.html
```

시스템 구조와 데이터 흐름을 시각적으로 확인할 수 있습니다.

---

## 파인튜닝 모델 API 정보

**API 엔드포인트:** `https://csmart-ai-faq-finetuning.hf.space/predict`

**사용된 모델:** Gemma 2B (파인튜닝)

**특징:**
- 빠른 응답 속도
- 일반적인 학습 조언에 최적화
- 자동 재시도 (최대 3회)
- 타임아웃 120초
- LLM 재가공으로 답변 품질 향상
- 품질 평가 후 필요시 LangGraph로 자동 재라우팅

---

## 시스템 아키텍처

자세한 시스템 구조와 데이터 흐름은 다음 파일들을 참고하세요:

- `system_architecture.md`: 상세한 아키텍처 문서
- `generate_diagrams.html`: 시각적 다이어그램 (브라우저에서 확인)

### 핵심 컴포넌트
1. **질문 복잡도 판별기**: LLM 기반 자동 분류
2. **파인튜닝 모델**: 빠른 일반 질문 처리
3. **답변 재가공기**: LLM 기반 답변 품질 향상
4. **품질 평가기**: 답변 품질 자동 평가
5. **LangGraph 에이전트**: 복잡한 질문 처리
6. **하이브리드 검색**: GuidelineDB + 웹 검색


