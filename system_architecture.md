# CSmart 시스템 아키텍처

## 전체 시스템 구조

```mermaid
graph TD
    A[사용자 질문] --> B{질문 복잡도 판별}
    
    B -->|간단한 질문| C[파인튜닝 모델]
    B -->|복잡한 질문| D[LangGraph 에이전트]
    
    C --> C1[CSmart-FAQ API 호출]
    C1 --> C2[Gemma 2B 파인튜닝 모델]
    C2 --> C3[원시 답변 생성]
    C3 --> C4[LLM 재가공]
    C4 --> C5[재가공된 답변]
    C5 --> C6{답변 품질 평가}
    
    C6 -->|품질 양호 6점 이상| C7[재가공된 답변 반환]
    C6 -->|품질 미달 6점 미만| C8[LangGraph로 재라우팅]
    
    C8 --> D
    D --> D1[통합 에이전트]
    D1 --> D2{데이터 소스 선택}
    
    D2 -->|GuidelineDB| E[가이드라인 에이전트]
    D2 -->|Web Search| F[웹 검색 에이전트]
    
    E --> E1[하이브리드 검색]
    E1 --> E2[키워드 매칭]
    E1 --> E3[벡터 유사도]
    E2 --> E4[Chroma DB]
    E3 --> E4
    E4 --> E5[문서 검색]
    E5 --> E6[답변 생성]
    
    F --> F1[Tavily API]
    F1 --> F2[웹 검색]
    F2 --> F3[검색 결과]
    F3 --> F4[답변 생성]
    
    E6 --> G[최종 답변]
    F4 --> G
    C7 --> G
    
    G --> H[사용자에게 응답]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e8
    style C4 fill:#e8f5e8
    style C6 fill:#fff3e0
    style C8 fill:#ffebee
    style D fill:#f3e5f5
    style G fill:#fff9c4
    style H fill:#e1f5fe
```

## 라우팅 로직 상세

```mermaid
flowchart TD
    A[질문 입력] --> B{수동선택 여부}
    B -->|Yes| B1{어떤 모델?}
    B1 -->|파인튜닝 모델| C[파인튜닝 모델 호출]
    B1 -->|LangGraph 에이전트| D[LangGraph 에이전트 호출]
    
    B -->|No| E[질문 복잡도 판별]
    E --> F{질문 분석}
    F -->|일반적인 학습 조언| G[Simple 분류]
    F -->|특정 대학/일정 정보| H[Complex 분류]
    
    G --> C
    H --> D
    
    C --> C1[파인튜닝 모델 원시 답변]
    C1 --> C2[LLM 재가공]
    C2 --> C3[재가공된 답변]
    C3 --> C4{답변 품질 평가}
    C4 -->|6점 이상| C5[재가공된 답변 반환]
    C4 -->|6점 미만| C6[LangGraph로 재라우팅]
    C6 --> D
    
    D --> J[검색 기반 답변 생성]
    
    C5 --> K[최종 답변]
    J --> K
```

## 데이터 흐름도

```mermaid
sequenceDiagram
    participant U as 사용자
    participant API as API Layer
    participant R as Router
    participant F as 파인튜닝 모델
    participant RF as 답변 재가공
    participant E as 답변 품질 평가
    participant L as LangGraph
    participant DB as Chroma DB
    participant W as Web Search
    
    U->>API: 질문 입력
    API->>R: 질문 복잡도 판별
    R->>R: LLM으로 분석
    
    alt 간단한 질문
        R->>F: CSmart-FAQ API 호출
        F->>F: Gemma 2B 모델 처리
        F->>R: 원시 답변 반환
        R->>RF: 답변 재가공 요청
        RF->>RF: LLM으로 답변 개선
        RF->>R: 재가공된 답변 반환
        R->>E: 답변 품질 평가 요청
        E->>E: LLM으로 품질 분석 (1-10점)
        
        alt 품질 양호 (6점 이상)
            E->>R: 품질 양호 판정
            R->>API: 재가공된 답변 전달
        else 품질 미달 (6점 미만)
            E->>R: 품질 미달 판정
            R->>L: LangGraph로 재라우팅
            L->>DB: 하이브리드 검색
            L->>W: 웹 검색 (필요시)
            DB->>L: 관련 문서 반환
            W->>L: 웹 검색 결과
            L->>L: 답변 생성
            L->>R: 생성된 답변
            R->>API: LangGraph 답변 전달
        end
    else 복잡한 질문
        R->>L: LangGraph 에이전트 실행
        L->>DB: 하이브리드 검색
        L->>W: 웹 검색 (필요시)
        DB->>L: 관련 문서 반환
        W->>L: 웹 검색 결과
        L->>L: 답변 생성
        L->>R: 생성된 답변
        R->>API: 답변 전달
    end
    
    API->>U: 최종 답변 반환
```

## 컴포넌트별 상세 구조

```mermaid
graph LR
    subgraph "API Layer"
        A1[get_answer 함수]
        A2[질문 복잡도 판별]
        A3[파인튜닝 모델 호출]
        A4[답변 재가공]
        A5[답변 품질 평가]
    end
    
    subgraph "파인튜닝 모델"
        F1[CSmart-FAQ API]
        F2[Gemma 2B 모델]
        F3[답변 생성]
    end
    
    subgraph "LangGraph 시스템"
        L1[통합 에이전트]
        L2[가이드라인 에이전트]
        L3[웹 검색 에이전트]
    end
    
    subgraph "데이터 소스"
        D1[Chroma DB]
        D2[Tavily API]
        D3[Google Gemini]
    end
    
    A1 --> A2
    A1 --> A3
    A1 --> A4
    A1 --> A5
    A2 --> L1
    A3 --> F1
    A4 --> A5
    A5 --> L1
    F1 --> F2
    F2 --> F3
    L1 --> L2
    L1 --> L3
    L2 --> D1
    L3 --> D2
    D1 --> D3
```

## 답변 처리 방식 비교

```mermaid
graph TD
    A[질문 입력] --> B{질문 유형}
    
    B -->|간단한 질문| C[파인튜닝 모델]
    B -->|복잡한 질문| D[LangGraph 에이전트]
    
    C --> C1[API 호출]
    C1 --> C2[모델 처리]
    C2 --> C3[원시 답변 생성]
    C3 --> C4[LLM 재가공]
    C4 --> C5[재가공된 답변]
    C5 --> C6{답변 품질 평가}
    C6 -->|6점 이상| C7[재가공된 답변 반환]
    C6 -->|6점 미만| C8[LangGraph로 재라우팅]
    C8 --> D
    
    D --> D1[데이터 검색]
    D1 --> D2[문서 분석]
    D2 --> D3[답변 생성]
    D3 --> D4[검색 기반 답변]
    
    C7 --> E[최종 답변]
    D4 --> E
    
    style C4 fill:#e8f5e8
    style C6 fill:#fff3e0
    style C8 fill:#ffebee
    style D4 fill:#e8f5e8
```

## 주요 특징

### 1. 자동 라우팅
- 질문 복잡도에 따른 자동 모델 선택
- LLM 기반 지능형 분류

### 2. 이중 시스템
- **간단한 질문**: 파인튜닝 모델 (빠른 응답)
- **복잡한 질문**: LangGraph 에이전트 (상세한 답변)

### 3. 답변 처리 방식
- **파인튜닝 모델**: 원시 답변 생성 → LLM 재가공 → 답변 품질 평가 후 6점 이상이면 그대로 반환, 미달 시 LangGraph로 재라우팅
- **LangGraph**: 검색 기반 답변 생성 및 재가공

### 4. 확장성
- 새로운 데이터 소스 추가 가능
- 에이전트 모듈화 설계

### 5. 오류 처리
- 각 단계별 오류 처리
- 파인튜닝 모델 오류 시 LangGraph로 자동 폴백
- 답변 품질 미달 시 자동 재라우팅
