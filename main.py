"""
CSmart 편입 상담 FastAPI 서버
LangGraph AI 에이전트를 FastAPI로 래핑하여 REST API 제공
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uvicorn
import os

# 기존 API 모듈 import
from api import get_answer

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="CSmart 편입 상담 AI API",
    description="편입 상담을 위한 AI 에이전트 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 스키마 정의
class StudentProfile(BaseModel):
    target_university: str = Field(default="미지정", description="목표 대학")
    track: str = Field(default="계열 미지정", description="계열 (이과/문과)")

class Dialogue(BaseModel):
    role: str = Field(..., description="역할 (student/teacher)")
    message: str = Field(..., description="메시지 내용")

class ChatRequest(BaseModel):
    question: str = Field(..., description="학생의 질문")
    student_profile: Optional[StudentProfile] = Field(default=None, description="학생 프로필")
    recent_dialogues: Optional[List[Dialogue]] = Field(default=[], description="최근 대화 내역")

class ChatResponse(BaseModel):
    question: str
    final_answer: str
    context: str
    datasources: List[str]
    success: bool
    error: Optional[str] = None

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "CSmart 편입 상담 AI API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "service": "csmart-langraph",
        "version": "1.0.0"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    편입 상담 질문 처리
    
    Args:
        request: 채팅 요청 데이터
        
    Returns:
        ChatResponse: AI 답변 결과
    """
    try:
        # 학생 프로필 변환
        student_profile = None
        if request.student_profile:
            student_profile = {
                "target_university": request.student_profile.target_university,
                "track": request.student_profile.track
            }
        
        # 대화 내역 변환
        recent_dialogues = []
        if request.recent_dialogues:
            recent_dialogues = [
                {
                    "role": dialogue.role,
                    "message": dialogue.message
                }
                for dialogue in request.recent_dialogues
            ]
        
        # AI 에이전트 실행
        result = get_answer(
            question=request.question,
            student_profile=student_profile,
            recent_dialogues=recent_dialogues,
            verbose=False  # API에서는 로그 최소화
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI 에이전트 실행 중 오류가 발생했습니다: {str(e)}"
        )

@app.get("/api/status")
async def get_status():
    """서비스 상태 및 정보 조회"""
    return {
        "service": "CSmart LangGraph API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/health",
            "status": "/api/status"
        }
    }

if __name__ == "__main__":
    # 환경 변수에서 포트 설정 (기본값: 8000)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # 프로덕션에서는 False
        log_level="info"
    )
