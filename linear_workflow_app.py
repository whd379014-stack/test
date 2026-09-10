import io
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from docx import Document


st.set_page_config(
    page_title="Linear LLM Workflow Studio",
    page_icon="🔗",
    layout="wide",
)


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_MODELS = [
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
]
EMBEDDING_MODEL = "text-embedding-3-small"


SELF_INTRO_USER_TEMPLATE = """[회사명]
예: 세아특수강

[지원 직무]
예: 생산기술 / 설비기술 / 품질

[자기소개서 문항]
문항을 그대로 붙여 넣어 주세요.

[글자 수 제한]
예: 700자 (공백 포함) / 제한 없음 / 잘 모르겠음

[현재 초안]
없으면 '초안 없음'이라고 적고, 아래 사실만 입력해도 됩니다.

[반드시 유지해야 할 사실 / 경험]
- 역할:
- 상황/문제:
- 내가 직접 한 행동:
- 수치/성과:
- 배운 점:

[기업·직무 관련 정보]
알고 있는 사실을 적거나, 03 기업·직무 맞춤 편집자에게 채용공고/기업자료를 RAG로 업로드해 주세요.

[원하는 수정 방향]
예: 두괄식, STAR, 직무 연관성 강화, 과장 없이, 소제목 포함
"""


SELF_INTRO_PRESET = [
    {
        "role": "intent",
        "name": "01 문항의도·소재 선별관",
        "description": "채용담당자 관점에서 질문의 평가 의도를 해석하고, 가장 강한 경험과 핵심 메시지를 선별합니다.",
        "system_prompt": """당신은 대기업/중견기업 신입 채용을 오래 담당한 채용담당자이자 자기소개서 평가자다.

목표는 최종 문장을 대신 써주는 것이 아니라, 다음 에이전트가 정확하게 첨삭할 수 있도록 '질문의 의도와 쓸 재료'를 잠그는 것이다. 입력에는 회사명, 직무, 문항, 글자 수, 초안, 경험 사실, RAG 자료가 섞여 있을 수 있다.

반드시 지킬 원칙:
1) 문항을 지원동기/직무역량/문제해결/협업·리더십/도전·실패/성장/입사 후 포부/자유문항 중 가장 가까운 유형으로 분류한다.
2) 질문이 실제로 평가하려는 역량과 반드시 답해야 하는 요소를 먼저 제시한다.
3) 사용자가 제공한 경험 중 문항과 직무에 가장 직접적인 1개 또는 최대 2개만 선택한다. 관련 없는 경험 나열은 제거 대상으로 표시한다.
4) 추상적인 성격 주장보다 구체적인 행동, 개인의 역할, 검증 가능한 결과를 우선한다.
5) 사용자가 말하지 않은 수치, 성과, 직무 경험, 회사 정보는 절대 만들지 않는다. 부족한 사실은 [확인 필요]로 표시한다.
6) 팀 경험은 '우리 팀이 했다'가 아니라 사용자가 직접 맡은 역할과 행동이 드러나도록 분석한다.
7) 지원동기는 회사 고유의 기술/고객/제품·서비스/일하는 방식과 사용자의 경험이 연결되는지 본다. 단순한 유명세, 성장성, 복지 선호는 핵심 근거로 사용하지 않는다.
8) 입사 후 포부는 막연한 미래 약속이 아니라 실제 맡게 될 업무 이해가 드러나는지 본다.

출력 형식:
[문항 유형]
[채용담당자가 보는 핵심]
[한 문장 핵심 메시지]
[살릴 경험과 근거]
[버리거나 줄일 내용]
[확인 필요한 사실]
[초안의 가장 큰 위험 3가지]
[다음 단계 전달용 SOURCE PACKET]
- 회사/직무/문항/글자수
- 사용자가 제공한 원문 초안
- 확인된 경험 사실과 수치
- 절대로 새로 만들면 안 되는 정보

SOURCE PACKET에서는 사용자가 준 핵심 사실과 원문 초안을 누락하지 말고 다음 에이전트가 재작성할 수 있게 보존하라.""",
        "chat_starter": """아래 자기소개서 문항과 초안을 채용담당자 관점에서만 진단해줘. 아직 최종본은 쓰지 말고, 질문의 의도·핵심 메시지·살릴 경험·버릴 내용·추가로 확인할 사실을 정리해줘.

[문항]

[초안]
""",
        "workflow_prompt": "사용자 입력을 분해해 질문의 평가 의도와 핵심 메시지를 먼저 확정하라. 원문 초안과 확인된 사실은 SOURCE PACKET에 반드시 보존하고, 추정이나 새로운 수치를 만들지 마라.",
        "rag_hint": "추천 RAG: 채용공고, 직무기술서, 인재상, 본인 경험정리/이력서. 문항 의도 판단에 실제 공고 문구가 있으면 더 정확합니다.",
    },
    {
        "role": "star",
        "name": "02 STAR·팩트 구조설계자",
        "description": "선별된 경험을 두괄식과 STAR/CAR 구조로 재배치하면서 개인 기여도와 팩트를 지킵니다.",
        "system_prompt": """당신은 자기소개서 구조 설계 전문 에디터다. 이전 에이전트의 SOURCE PACKET을 기반으로 초안을 '읽히는 구조'로 바꾼다.

반드시 지킬 원칙:
1) 첫 문장 또는 첫 문단에서 문항에 대한 답과 지원자의 강점을 먼저 제시하는 두괄식을 사용한다.
2) 경험형 문항은 STAR 또는 CAR 구조를 사용하되 S/T는 짧게, A는 가장 구체적으로, R은 검증 가능한 결과와 배운 점으로 마무리한다.
3) '우리 팀이'보다 '나는 무엇을 맡아 무엇을 바꿨는가'가 드러나야 한다.
4) 수치는 SOURCE PACKET 또는 RAG에 존재하는 것만 사용한다. 수치가 없으면 억지로 만들지 말고 행동의 범위·빈도·대상·전후 차이를 구체화한다.
5) 지원동기와 입사 후 포부는 억지로 STAR에 끼워 넣지 않는다.
   - 지원동기: 회사/직무의 고유한 이유 → 나의 관련 경험/관심 근거 → 입사 후 기여 가능성
   - 입사 후 포부: 입사 직후 실제 업무 → 이를 익힐 나의 경험 기반 → 이후 도전 영역/회사 방향과의 연결
6) 문항과 무관한 배경 설명, 감정 과잉, 상투적인 교훈은 줄인다.
7) 사실을 새로 만들거나 회사 정보를 추정하지 않는다.

출력 형식:
[구조 설계]
- 핵심 주장
- S/C 상황
- T 과제
- A 행동(가장 자세히)
- R 결과/학습

[개선 초안 v1]
문항에 바로 제출할 수 있는 자연스러운 한국어 초안을 작성한다.

[팩트 잠금]
- 유지한 수치/고유명사/역할
- [확인 필요] 항목""",
        "chat_starter": """이 경험을 사실 추가 없이 두괄식 + STAR 구조로 재구성해줘. 특히 내가 직접 한 행동과 결과가 가장 많이 보이게 해줘. 수치는 새로 만들지 마.

[문항]

[경험 사실]

[현재 초안]
""",
        "workflow_prompt": "1단계 SOURCE PACKET에 있는 사실만 사용해 두괄식 구조와 STAR/CAR 흐름을 설계하고 개선 초안 v1을 작성하라. 지원동기/입사 후 포부라면 해당 문항에 더 적합한 구조를 우선하라.",
        "rag_hint": "추천 RAG: 이력서, 프로젝트 정리, 경험기술서, 기존 자소서. 본인 경험의 사실 보존이 가장 중요합니다.",
    },
    {
        "role": "fit",
        "name": "03 기업·직무 맞춤 편집자",
        "description": "채용공고와 기업 자료를 근거로 복붙형 문장을 제거하고 회사·직무 고유성을 강화합니다.",
        "system_prompt": """당신은 기업/직무 맞춤형 자기소개서 편집자다. 이전 초안의 좋은 구조는 유지하면서 '왜 이 회사, 왜 이 직무인지'를 근거 있게 강화한다.

반드시 지킬 원칙:
1) RAG 또는 사용자가 제공한 자료에 근거한 회사/직무 정보만 사용한다. 근거가 없는 최신 사업, 기술, 고객, 수치, 전략을 만들어내지 않는다.
2) 채용공고의 실제 주요업무, 우대사항, 필요역량이 있으면 자기소개서의 행동·역량·포부와 직접 연결한다.
3) 지원동기에서는 아래 연결고리 중 근거가 강한 것만 선택한다: 회사 고유 기술 경쟁력, 고객 문제 이해, 제품/서비스의 차별점, 직무 특수성/일하는 방식.
4) '성장하는 회사라서', '유명해서', '비전에 공감해서', '회사와 함께 성장'처럼 어느 회사에도 붙는 문장은 구체 근거가 없으면 삭제/교체한다.
5) 입사 후 포부라면 '입사 직후 실제 맡을 업무 → 사용자의 관련 경험을 토대로 익힐 방법 → 이후 도전할 영역 → 회사가 실제 추진 중인 방향' 순서가 자연스러운지 점검한다. 회사 방향이 자료에 없으면 억지로 넣지 말고 [회사 자료 필요]로 표시한다.
6) 전문용어는 실제 직무 맥락에 맞고 사용자가 이해한 범위에서만 쓴다. 과도한 용어 나열은 금지한다.
7) 사용자 경험의 사실과 수치는 변경하지 않는다.

출력 형식:
[직무·기업 맞춤 판단]
- 현재 고유성 수준: 상/중/하
- 살릴 회사/직무 근거
- 제거할 복붙형 표현
- [회사 자료 필요] 항목

[개선 초안 v2]

[사용한 근거]
사용자 입력/RAG에서 실제로 확인된 근거만 짧게 정리한다.""",
        "chat_starter": """채용공고/기업자료를 근거로 아래 초안을 이 회사와 이 직무에만 통하는 글로 바꿔줘. 자료에 없는 회사 사실은 절대 추가하지 말고, 부족하면 무엇이 필요한지 알려줘.

[회사/직무]

[문항]

[초안]
""",
        "workflow_prompt": "이전 초안을 채용공고·기업자료와 대조해 회사/직무 고유성을 강화하라. RAG 또는 입력에서 확인되지 않은 회사 사실은 쓰지 말고, 복붙 가능한 문장을 우선 제거하라.",
        "rag_hint": "강력 추천 RAG: 지원 회사 채용공고, 직무 소개, 기업 IR/사업자료 + 이번에 올린 `!입사 후 포부를 말해주세요.pdf`, `!골라쓰는 자소서 지원동기 5가지.pdf`. 이 에이전트가 가장 RAG 효과가 큽니다.",
    },
    {
        "role": "audit",
        "name": "04 금지어·논리 감사관",
        "description": "모호함, 과장, 복붙형 문장, 금지표현, 직무 무관 내용과 논리 비약을 찾아 수정합니다.",
        "system_prompt": """당신은 깐깐한 자기소개서 품질감사관이다. 좋은 내용을 새로 발명하는 것이 아니라, 이전 초안의 감점 요소를 찾아 사실을 유지한 채 교정한다.

검수 규칙:
1) 모호한 표현: '열심히', '최선을 다해', '다양한 경험', '많이 배웠다', '성장했다'처럼 근거 없는 추상어를 구체 행동/근거로 바꾼다.
2) 실패/어려움은 변명이나 타인 탓이 아니라 대응 행동과 개선으로 서술한다.
3) 직무와 무관한 경험 나열, 지나치게 사적인 정보, 과장/거짓 가능성이 있는 표현을 제거한다.
4) 선천성 강조('타고난', '천부적'), 지나친 줄임말, '저희/우리 팀'만 강조해 개인 기여가 사라지는 문장을 피한다.
5) '~라고 생각합니다' 반복을 줄이고, 가능한 곳은 사실/행동 중심의 단정적인 문장으로 바꾼다.
6) '귀사', '기회를 주신다면', '뽑아만 주신다면' 같은 거리감 있거나 구걸식 표현을 피한다. 회사명이나 직무 맥락을 자연스럽게 사용한다.
7) '빠르게 배우겠습니다', '회사와 함께 성장하겠습니다', '5년 후 전문가/10년 후 리더'처럼 구체 업무가 없는 포부는 복붙형으로 판단한다.
8) 수치와 인과관계가 입력 자료로 뒷받침되는지 확인한다. 근거 없는 수치/성과는 삭제하거나 [확인 필요]로 표시한다.
9) 문장은 간결하게, 한 문장에 메시지 하나를 원칙으로 한다.

출력 형식:
[감점 위험 검수표]
- 문제 표현 | 이유 | 수정 방향

[개선 초안 v3]
감점 요소를 제거한 전체 초안.

[남은 확인 필요]
사실 확인이 필요한 부분만 남긴다.""",
        "chat_starter": """아래 자소서를 채용담당자가 감점할 만한 표현 위주로 감사해줘. 모호함/과장/금지어/우리 팀 중심 표현/생각합니다 반복/복붙형 포부/직무 무관 내용을 표시하고 바로 고친 버전도 줘.

[초안]
""",
        "workflow_prompt": "이전 초안을 금지표현·모호함·과장·복붙 가능성·개인 기여도·논리 비약 기준으로 감사하고, 사실은 바꾸지 않은 개선 초안 v3을 작성하라.",
        "rag_hint": "추천 RAG: 이번에 올린 `!자소서 쓸 때 반드시 피해야 하는 5가지.pdf`, `!자기소개서 금지어 8가지.pdf`. 시스템 프롬프트에도 핵심 규칙이 들어 있어 RAG 없이도 동작합니다.",
    },
    {
        "role": "headline",
        "name": "05 소제목·두괄식 카피라이터",
        "description": "검증된 사실에서만 소제목 후보를 만들고, 첫 문장을 두괄식으로 날카롭게 다듬습니다.",
        "system_prompt": """당신은 자기소개서 소제목과 첫 문장 전문 카피라이터다. 화려한 문구보다 지원자의 실제 성과와 문제해결 방식이 한눈에 들어오게 만든다.

반드시 지킬 원칙:
1) 소제목은 본문에 실제로 존재하는 사실만 사용한다. 없는 수치, 성과, 별명, 기술명을 새로 만들지 않는다.
2) 가능하면 다음 유형을 고르게 제안한다: 수치/성과형, 문제해결형, 프로젝트·실행형, 리더십·협업형, 직무전문성형.
3) 제목만 보고 본문의 핵심 행동 또는 결과가 예상되어야 한다. 과장된 광고 문구, 억지 비유, 감성적인 명언형 제목은 피한다.
4) 숫자가 실제로 있을 때만 숫자를 쓴다. 숫자가 없으면 구체 행동/변화를 사용한다.
5) 첫 문장은 문항의 답 또는 핵심 역량을 바로 말하는 두괄식으로 다듬는다.
6) 제목과 첫 문장이 같은 말을 반복하지 않게 한다.
7) 글자 수 제한이 빡빡하면 소제목은 짧게 하고 본문 정보량을 우선한다.

출력 형식:
[소제목 후보 TOP 5]
1. ... (유형)
...

[최종 추천 소제목]

[두괄식 첫 문장]

[개선 초안 v4]
최종 추천 소제목을 포함해 전체 글을 다시 정리한다.""",
        "chat_starter": """이 자소서 본문에 실제로 있는 사실만 이용해 소제목 5개를 제안해줘. 성과형/문제해결형/직무강점형을 섞고, 없는 숫자는 만들지 마. 그중 1개를 골라 두괄식 첫 문장까지 다듬어줘.

[초안]
""",
        "workflow_prompt": "본문에 존재하는 사실만 사용해 소제목 5개를 만들고 하나를 선택하라. 없는 수치나 성과를 만들지 말고, 첫 문장은 문항에 대한 답이 바로 보이도록 두괄식으로 다듬은 개선 초안 v4를 작성하라.",
        "rag_hint": "추천 RAG: 이번에 올린 `!인사팀이 원하는 소제목 예시 모음집.pdf`, `!자기소개서 소제목 샘플 (ver. 수치화성과).pdf`. 예시는 패턴만 참고하고 문구를 그대로 베끼지 않도록 합니다.",
    },
    {
        "role": "final",
        "name": "06 최종 HR 제출본 에디터",
        "description": "앞 단계 결과를 한 번 더 채용담당자 시선으로 압축해 바로 제출 가능한 최종본으로 만듭니다.",
        "system_prompt": """당신은 마지막 제출 직전의 자기소개서 최종 편집자다. 앞선 에이전트가 검증한 사실, 구조, 회사/직무 연결, 감점 요소 수정, 소제목을 통합해 '바로 제출 가능한 한 개의 최종본'을 만든다.

최종 체크 기준:
- 문항에 첫 부분부터 직접 답하는가
- 핵심 메시지가 하나로 모이는가
- 지원자의 개인 역할과 행동이 구체적인가
- 성과/수치/회사 정보는 제공된 근거 안에만 있는가
- 직무와 회사 연결이 자연스럽고 복붙형이 아닌가
- 불필요한 감정, 미사여구, 반복, 금지표현이 없는가
- 문단 간 인과가 자연스러운가
- 사용자가 요구한 두괄식/STAR/소제목 선호가 반영되었는가
- 글자 수 제한이 있으면 가능한 한 제한 안에서 정보 밀도를 높였는가

절대 규칙:
1) 마지막 단계에서 새로운 사실, 수치, 수상, 직무 경험, 회사 정보를 추가하지 않는다.
2) 사실이 불명확하면 매끈하게 지어내지 말고 해당 내용을 빼거나 최소화한다.
3) 사용자의 말투를 지나치게 인위적인 취업 컨설팅 문체로 바꾸지 않는다.
4) 최종 제출본은 읽기 쉬운 한국어로 작성한다.

출력 형식:
### 최종 제출본
(소제목 포함, 그대로 복사 가능한 본문)

### 최종 점검
- 문항 적합도: /10
- 구체성: /10
- 직무·기업 적합도: /10
- 차별성: /10
- 진정성/팩트 안전성: /10
- 가장 크게 좋아진 점 2개
- 사용자가 최종 확인해야 할 사실이 있으면 1~3개, 없으면 '없음'

점수는 후하게 주지 말고 실제 채용담당자 관점에서 짧게 평가하라.""",
        "chat_starter": """아래 자소서를 최종 제출본 수준으로만 다듬어줘. 새로운 사실은 추가하지 말고, 두괄식·개인 기여·직무 적합성·가독성·글자수 제한을 마지막으로 확인해줘. 최종본을 맨 위에 줘.

[문항]

[글자수]

[초안]
""",
        "workflow_prompt": "앞 단계의 검증된 사실만 사용해 바로 제출 가능한 최종본을 작성하라. 새 사실은 절대 만들지 말고, 문항 적합성·구체성·개인 기여·직무/기업 연결·가독성·글자수 제한을 최종 점검하라.",
        "rag_hint": "RAG는 보통 불필요합니다. 앞 단계가 충분한 근거를 전달하도록 두는 편이 최종 문장 품질이 안정적입니다.",
    },
]


# -----------------------------
# Session state
# -----------------------------
def init_state() -> None:
    defaults = {
        "agents": {},
        "workflow": [],
        "workflow_prompts": {},
        "chat_histories": {},
        "workflow_runs": [],
        "api_key": "",
        "api_ok": False,
        "next_agent_num": 1,
        "run_prompt": "",
        "self_intro_preset_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# -----------------------------
# Utilities
# -----------------------------
def get_client() -> OpenAI:
    key = st.session_state.api_key.strip()
    if not key:
        raise ValueError("OpenAI API key를 먼저 입력해 주세요.")
    return OpenAI(api_key=key)


def new_agent() -> Dict[str, Any]:
    n = st.session_state.next_agent_num
    st.session_state.next_agent_num += 1
    agent_id = f"agent_{n}"
    return {
        "id": agent_id,
        "name": f"Agent {n}",
        "model": DEFAULT_MODEL,
        "custom_model": "",
        "system_prompt": "You are a helpful AI agent. Follow the user's instructions accurately.",
        "rag_enabled": False,
        "rag_top_k": 4,
        "rag_chunks": [],
        "rag_embeddings": None,
        "rag_files": [],
        "rag_fingerprint": "",
    }


def load_self_intro_preset() -> List[str]:
    """Create or reuse the self-introduction editing agents and set them as the active workflow."""
    existing = {}
    for aid, agent in st.session_state.agents.items():
        if agent.get("preset_group") == "self_intro_editor_v1":
            existing[agent.get("preset_role")] = aid

    workflow_ids: List[str] = []
    for spec in SELF_INTRO_PRESET:
        role = spec["role"]
        if role in existing and existing[role] in st.session_state.agents:
            aid = existing[role]
            agent = st.session_state.agents[aid]
        else:
            agent = new_agent()
            aid = agent["id"]
            st.session_state.agents[aid] = agent
            st.session_state.chat_histories[aid] = []

        agent.update({
            "name": spec["name"],
            "model": DEFAULT_MODEL,
            "custom_model": "",
            "system_prompt": spec["system_prompt"],
            "chat_starter": spec["chat_starter"],
            "description": spec["description"],
            "rag_hint": spec["rag_hint"],
            "preset_group": "self_intro_editor_v1",
            "preset_role": role,
        })
        workflow_ids.append(aid)
        st.session_state.workflow_prompts[aid] = spec["workflow_prompt"]

    st.session_state.workflow = workflow_ids
    if not st.session_state.run_prompt.strip():
        st.session_state.run_prompt = SELF_INTRO_USER_TEMPLATE
    st.session_state.self_intro_preset_loaded = True
    return workflow_ids


def effective_model(agent: Dict[str, Any]) -> str:
    custom = agent.get("custom_model", "").strip()
    return custom if custom else agent.get("model", DEFAULT_MODEL)


def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)

    if name.endswith((".txt", ".md", ".csv")):
        for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    raise ValueError(f"지원하지 않는 파일 형식입니다: {uploaded_file.name}")


def chunk_text(text: str, source: str, chunk_size: int = 1400, overlap: int = 220) -> List[Dict[str, str]]:
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []

    chunks = []
    start = 0
    idx = 1
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append({"source": source, "chunk_id": idx, "text": piece})
            idx += 1
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def file_fingerprint(files) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode("utf-8", errors="ignore"))
        h.update(f.getvalue())
    return h.hexdigest()


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    vectors = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in result.data)
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def build_rag_index(agent: Dict[str, Any], files) -> None:
    client = get_client()
    chunks: List[Dict[str, str]] = []
    filenames = []

    for f in files:
        text = extract_text_from_file(f)
        filenames.append(f.name)
        chunks.extend(chunk_text(text, f.name))

    if not chunks:
        raise ValueError("파일에서 검색 가능한 텍스트를 찾지 못했습니다.")

    embeddings = embed_texts(client, [c["text"] for c in chunks])
    agent["rag_chunks"] = chunks
    agent["rag_embeddings"] = embeddings
    agent["rag_files"] = filenames
    agent["rag_fingerprint"] = file_fingerprint(files)


def retrieve_context(agent: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    if not agent.get("rag_enabled"):
        return []
    chunks = agent.get("rag_chunks", [])
    embeddings = agent.get("rag_embeddings")
    if not chunks or embeddings is None:
        return []

    client = get_client()
    q = client.embeddings.create(model=EMBEDDING_MODEL, input=[query]).data[0].embedding
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm != 0:
        q = q / norm

    scores = embeddings @ q
    top_k = min(int(agent.get("rag_top_k", 4)), len(chunks))
    idxs = np.argsort(scores)[-top_k:][::-1]
    results = []
    for i in idxs:
        item = dict(chunks[int(i)])
        item["score"] = float(scores[int(i)])
        results.append(item)
    return results


def make_user_input(base_input: str, rag_hits: List[Dict[str, Any]]) -> str:
    if not rag_hits:
        return base_input

    context_blocks = []
    for hit in rag_hits:
        context_blocks.append(
            f"[Source: {hit['source']} | chunk {hit['chunk_id']}]\n{hit['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return (
        "Use the retrieved reference context below when it is relevant. "
        "If the context does not support a claim, do not invent it.\n\n"
        f"<retrieved_context>\n{context}\n</retrieved_context>\n\n"
        f"<user_input>\n{base_input}\n</user_input>"
    )


def call_agent(agent: Dict[str, Any], user_input: str, history: List[Dict[str, str]] | None = None):
    client = get_client()
    rag_hits = retrieve_context(agent, user_input)
    final_user_input = make_user_input(user_input, rag_hits)

    if history:
        input_items = []
        for msg in history:
            input_items.append({"role": msg["role"], "content": msg["content"]})
        input_items.append({"role": "user", "content": final_user_input})
    else:
        input_items = final_user_input

    response = client.responses.create(
        model=effective_model(agent),
        instructions=agent.get("system_prompt", ""),
        input=input_items,
        store=False,
    )
    return response.output_text, rag_hits


def extract_final_submission(text: str) -> str:
    """Extract only the copy-ready final answer when the final editor follows the preset format."""
    marker = "### 최종 제출본"
    check_marker = "### 최종 점검"
    if marker not in text:
        return text.strip()
    body = text.split(marker, 1)[1]
    if check_marker in body:
        body = body.split(check_marker, 1)[0]
    return body.strip()


def move_workflow_item(index: int, delta: int) -> None:
    new_index = index + delta
    wf = st.session_state.workflow
    if 0 <= new_index < len(wf):
        wf[index], wf[new_index] = wf[new_index], wf[index]


def remove_from_workflow(agent_id: str) -> None:
    st.session_state.workflow = [x for x in st.session_state.workflow if x != agent_id]
    st.session_state.workflow_prompts.pop(agent_id, None)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
    .small-muted {color:#777; font-size:0.88rem;}
    .workflow-box {
        border:1px solid rgba(128,128,128,.28); border-radius:12px;
        padding:12px 14px; margin:4px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Header / API key
# -----------------------------
st.title("🔗 Linear LLM Workflow Studio")
st.caption("에이전트를 만들고 → 개별 테스트하고 → 순서를 편집하고 → RAG와 함께 순차 실행하는 Streamlit 대시보드")

with st.sidebar:
    st.header("1) OpenAI 연결")
    st.session_state.api_key = st.text_input(
        "OpenAI API key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
        help="키는 Streamlit 세션 상태에만 보관되며 코드나 파일에 저장하지 않습니다.",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("키 확인", use_container_width=True):
            try:
                client = get_client()
                client.models.retrieve(DEFAULT_MODEL)
                st.session_state.api_ok = True
                st.success("연결 성공 · GPT-5.6 Luna 사용 가능")
            except Exception as e:
                st.session_state.api_ok = False
                st.error(f"연결 실패: {e}")
    with c2:
        if st.button("키 지우기", use_container_width=True):
            st.session_state.api_key = ""
            st.session_state.api_ok = False
            st.rerun()

    st.caption(
        "기본 실행 모델: GPT-5.6 Luna (`gpt-5.6-luna`). "
        "ChatGPT 구독과 OpenAI API의 사용 권한/결제는 별도로 관리됩니다."
    )
    st.divider()
    st.caption("RAG 지원 형식: PDF, DOCX, TXT, MD, CSV")
    st.caption("업로드 파일/인덱스/채팅은 현재 Streamlit 세션에만 유지됩니다.")


# -----------------------------
# Main tabs
# -----------------------------
tab_agents, tab_workflow, tab_run = st.tabs([
    "🤖 에이전트 만들기 & 채팅",
    "🧩 Linear Workflow 편집",
    "▶️ Workflow 실행",
])


# -----------------------------
# Agent builder & chat
# -----------------------------
with tab_agents:
    top_left, top_right = st.columns([1, 2])
    with top_left:
        st.subheader("새 에이전트")
        if st.button("＋ 에이전트 추가", type="primary", use_container_width=True):
            agent = new_agent()
            st.session_state.agents[agent["id"]] = agent
            st.session_state.chat_histories[agent["id"]] = []
            st.rerun()

        if st.button("✨ 자기소개서 첨삭 프리셋 불러오기", use_container_width=True):
            load_self_intro_preset()
            st.rerun()
        st.caption("프리셋: 문항 해석 → STAR 구조 → 기업·직무 맞춤 → 금지어 감사 → 소제목 → 최종 HR 편집")
        st.caption(f"현재 {len(st.session_state.agents)}개")

    with top_right:
        st.info(
            "새 에이전트의 기본 모델은 **GPT-5.6 Luna (`gpt-5.6-luna`)**입니다. "
            "**자기소개서 첨삭 프리셋**을 누르면 6명의 전문 에이전트와 Linear Workflow, "
            "각 단계 추가 프롬프트, 개별 채팅용 샘플 질문까지 자동으로 구성합니다."
        )
        if st.session_state.agents and st.button(
            "⚡ 모든 에이전트를 GPT-5.6 Luna로 변경",
            use_container_width=True,
        ):
            for _agent_id, _agent in st.session_state.agents.items():
                _agent["model"] = DEFAULT_MODEL
                _agent["custom_model"] = ""
                st.session_state[f"model_{_agent_id}"] = DEFAULT_MODEL
                st.session_state[f"custom_model_{_agent_id}"] = ""
            st.rerun()

    if not st.session_state.agents:
        st.warning("에이전트가 없습니다. 먼저 '에이전트 추가'를 눌러 주세요.")

    for agent_id in list(st.session_state.agents.keys()):
        agent = st.session_state.agents[agent_id]
        with st.expander(f"🤖 {agent['name']}  ·  {effective_model(agent)}", expanded=True):
            if agent.get("description"):
                st.caption(agent["description"])
            col_a, col_b = st.columns([2, 1])
            with col_a:
                agent["name"] = st.text_input("에이전트 이름", value=agent["name"], key=f"name_{agent_id}")
                agent["system_prompt"] = st.text_area(
                    "System Prompt",
                    value=agent["system_prompt"],
                    height=150,
                    key=f"sys_{agent_id}",
                )
            with col_b:
                agent["model"] = st.selectbox(
                    "기본 모델",
                    DEFAULT_MODELS,
                    index=DEFAULT_MODELS.index(agent["model"]) if agent["model"] in DEFAULT_MODELS else DEFAULT_MODELS.index(DEFAULT_MODEL),
                    key=f"model_{agent_id}",
                )
                agent["custom_model"] = st.text_input(
                    "Custom model ID (선택)",
                    value=agent.get("custom_model", ""),
                    placeholder="예: gpt-5.6-luna",
                    key=f"custom_model_{agent_id}",
                    help="입력하면 위 기본 모델보다 우선합니다.",
                )
                if st.button("에이전트 삭제", key=f"delete_{agent_id}", use_container_width=True):
                    st.session_state.agents.pop(agent_id, None)
                    st.session_state.chat_histories.pop(agent_id, None)
                    remove_from_workflow(agent_id)
                    st.rerun()

            st.markdown("#### 선택적 RAG")
            if agent.get("rag_hint"):
                st.info(agent["rag_hint"])
            rag_c1, rag_c2 = st.columns([1, 3])
            with rag_c1:
                agent["rag_enabled"] = st.toggle(
                    "RAG 사용",
                    value=agent.get("rag_enabled", False),
                    key=f"rag_toggle_{agent_id}",
                )
                agent["rag_top_k"] = st.slider(
                    "검색 chunk 수",
                    1,
                    8,
                    int(agent.get("rag_top_k", 4)),
                    key=f"topk_{agent_id}",
                )
            with rag_c2:
                uploads = st.file_uploader(
                    "참조 파일을 Drag & Drop",
                    type=["pdf", "docx", "txt", "md", "csv"],
                    accept_multiple_files=True,
                    key=f"files_{agent_id}",
                )
                if uploads:
                    current_fp = file_fingerprint(uploads)
                    is_stale = current_fp != agent.get("rag_fingerprint", "")
                    if is_stale:
                        st.warning("업로드 파일이 변경되었습니다. 아래 버튼으로 RAG 인덱스를 생성/갱신하세요.")
                if st.button("RAG 인덱스 생성 / 갱신", key=f"index_{agent_id}"):
                    if not uploads:
                        st.warning("먼저 파일을 업로드해 주세요.")
                    else:
                        try:
                            with st.spinner("문서를 읽고 임베딩 인덱스를 만드는 중..."):
                                build_rag_index(agent, uploads)
                            st.success(f"완료: {len(agent['rag_files'])}개 파일 / {len(agent['rag_chunks'])} chunks")
                        except Exception as e:
                            st.error(f"RAG 인덱스 생성 실패: {e}")
                if agent.get("rag_files"):
                    st.caption("현재 인덱스: " + ", ".join(agent["rag_files"]))

            st.markdown("#### 개별 에이전트 채팅")
            history = st.session_state.chat_histories.setdefault(agent_id, [])
            for msg in history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            quick_prompt = None
            if agent.get("chat_starter"):
                with st.expander("💬 이 에이전트 전용 샘플 프롬프트", expanded=False):
                    quick_prompt = st.text_area(
                        "샘플을 수정한 뒤 바로 테스트할 수 있습니다.",
                        value=agent["chat_starter"],
                        height=180,
                        key=f"quick_prompt_{agent_id}",
                    )
                    quick_clicked = st.button(
                        "이 프롬프트로 개별 채팅 실행",
                        key=f"quick_run_{agent_id}",
                        use_container_width=True,
                    )
                if quick_clicked and quick_prompt and quick_prompt.strip():
                    history.append({"role": "user", "content": quick_prompt.strip()})
                    try:
                        with st.spinner(f"{agent['name']} 응답 생성 중..."):
                            previous = history[:-1]
                            answer, hits = call_agent(agent, quick_prompt.strip(), previous)
                        history.append({"role": "assistant", "content": answer})
                        st.rerun()
                    except Exception as e:
                        st.error(f"API 호출 실패: {e}")

            chat_prompt = st.chat_input("이 에이전트에게 테스트 메시지 보내기", key=f"chat_input_{agent_id}")
            if chat_prompt:
                history.append({"role": "user", "content": chat_prompt})
                try:
                    with st.spinner(f"{agent['name']} 응답 생성 중..."):
                        previous = history[:-1]
                        answer, hits = call_agent(agent, chat_prompt, previous)
                    history.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"API 호출 실패: {e}")

            if st.button("채팅 초기화", key=f"clear_chat_{agent_id}"):
                st.session_state.chat_histories[agent_id] = []
                st.rerun()


# -----------------------------
# Workflow editor
# -----------------------------
with tab_workflow:
    st.subheader("Linear Workflow 편집")
    st.write("앞 단계의 **output이 다음 단계의 input**으로 전달됩니다. 각 단계에는 별도의 추가 프롬프트를 붙일 수 있습니다.")

    available = [aid for aid in st.session_state.agents if aid not in st.session_state.workflow]
    if available:
        label_to_id = {st.session_state.agents[aid]["name"]: aid for aid in available}
        add_col1, add_col2 = st.columns([3, 1])
        with add_col1:
            selected_name = st.selectbox("Workflow에 추가할 에이전트", list(label_to_id.keys()))
        with add_col2:
            st.write("")
            st.write("")
            if st.button("Workflow에 추가", use_container_width=True):
                aid = label_to_id[selected_name]
                st.session_state.workflow.append(aid)
                st.session_state.workflow_prompts.setdefault(aid, "")
                st.rerun()
    elif st.session_state.agents:
        st.caption("모든 에이전트가 Workflow에 포함되어 있습니다.")
    else:
        st.warning("먼저 에이전트를 생성해 주세요.")

    st.divider()

    if not st.session_state.workflow:
        st.info("Workflow가 비어 있습니다.")
    else:
        for idx, aid in enumerate(st.session_state.workflow):
            agent = st.session_state.agents.get(aid)
            if not agent:
                continue

            st.markdown(f"**STEP {idx + 1} → {agent['name']}**  ·  `{effective_model(agent)}`")
            st.caption("입력: " + ("사용자가 Workflow 실행 시 입력한 최초 prompt" if idx == 0 else "바로 앞 에이전트의 output"))

            c_prompt, c_buttons = st.columns([5, 1])
            with c_prompt:
                st.session_state.workflow_prompts[aid] = st.text_area(
                    "이 단계의 추가 프롬프트 (선택)",
                    value=st.session_state.workflow_prompts.get(aid, ""),
                    placeholder="예: 위 결과를 표 형태의 핵심 요약으로 재작성해줘.",
                    key=f"wf_prompt_{aid}",
                    height=90,
                )
            with c_buttons:
                if st.button("⬆ 위로", key=f"up_{aid}", disabled=(idx == 0), use_container_width=True):
                    move_workflow_item(idx, -1)
                    st.rerun()
                if st.button("⬇ 아래로", key=f"down_{aid}", disabled=(idx == len(st.session_state.workflow) - 1), use_container_width=True):
                    move_workflow_item(idx, 1)
                    st.rerun()
                if st.button("✕ 제거", key=f"rm_{aid}", use_container_width=True):
                    remove_from_workflow(aid)
                    st.rerun()

            if agent.get("rag_enabled"):
                if agent.get("rag_chunks"):
                    st.caption(f"📚 RAG ON · {len(agent['rag_files'])} files · top-{agent['rag_top_k']}")
                else:
                    st.warning("RAG가 켜져 있지만 인덱스가 없습니다. 에이전트 탭에서 파일을 업로드하고 인덱스를 생성하세요.")
            st.markdown("---")

        if st.button("Workflow 전체 비우기"):
            st.session_state.workflow = []
            st.session_state.workflow_prompts = {}
            st.rerun()


# -----------------------------
# Workflow runner
# -----------------------------
with tab_run:
    st.subheader("Workflow 실행")

    has_self_intro_workflow = any(
        st.session_state.agents.get(aid, {}).get("preset_group") == "self_intro_editor_v1"
        for aid in st.session_state.workflow
    )
    if has_self_intro_workflow:
        st.success("자기소개서 첨삭 Workflow가 활성화되어 있습니다. 아래 템플릿에 지원 정보를 채워 넣고 실행하세요.")
        if st.button("📝 자기소개서 입력 템플릿 다시 넣기"):
            st.session_state.run_prompt = SELF_INTRO_USER_TEMPLATE
            st.rerun()

    if st.session_state.workflow:
        names = [st.session_state.agents[aid]["name"] for aid in st.session_state.workflow if aid in st.session_state.agents]
        st.code("  →  ".join(names), language=None)
    else:
        st.warning("먼저 Linear Workflow를 구성해 주세요.")

    user_prompt = st.text_area(
        "최초 User Prompt",
        height=360 if has_self_intro_workflow else 150,
        placeholder="예: 회사명/직무/문항/글자수/현재 초안/경험 사실을 입력해 주세요.",
        key="run_prompt",
    )

    run_clicked = st.button(
        "▶ Workflow 실행",
        type="primary",
        disabled=(not st.session_state.workflow),
        use_container_width=True,
    )

    if run_clicked:
        if not user_prompt.strip():
            st.warning("User Prompt를 입력해 주세요.")
        else:
            current_input = user_prompt.strip()
            run_steps = []
            failed = False

            progress = st.progress(0)
            status = st.empty()

            for idx, aid in enumerate(st.session_state.workflow):
                agent = st.session_state.agents.get(aid)
                if not agent:
                    continue

                extra = st.session_state.workflow_prompts.get(aid, "").strip()
                if idx == 0:
                    step_input = current_input
                    if extra:
                        step_input += f"\n\n[Additional instruction for this step]\n{extra}"
                else:
                    step_input = (
                        "The following is the output from the previous agent. "
                        "Use it as the primary input for your task.\n\n"
                        f"<previous_agent_output>\n{current_input}\n</previous_agent_output>"
                    )
                    if extra:
                        step_input += f"\n\n[Additional instruction for this step]\n{extra}"

                status.info(f"STEP {idx + 1}/{len(st.session_state.workflow)} · {agent['name']} 실행 중")
                try:
                    output, rag_hits = call_agent(agent, step_input)
                    run_steps.append({
                        "step": idx + 1,
                        "agent_id": aid,
                        "agent_name": agent["name"],
                        "model": effective_model(agent),
                        "input": step_input,
                        "output": output,
                        "rag_hits": rag_hits,
                    })
                    current_input = output
                    progress.progress((idx + 1) / len(st.session_state.workflow))
                except Exception as e:
                    run_steps.append({
                        "step": idx + 1,
                        "agent_id": aid,
                        "agent_name": agent["name"],
                        "model": effective_model(agent),
                        "input": step_input,
                        "output": f"ERROR: {e}",
                        "rag_hits": [],
                    })
                    status.error(f"STEP {idx + 1} 실패: {e}")
                    failed = True
                    break

            if not failed:
                status.success("Workflow 실행 완료")

            st.session_state.workflow_runs.insert(0, {
                "user_prompt": user_prompt,
                "steps": run_steps,
                "final_output": current_input if not failed else None,
            })

    if st.session_state.workflow_runs:
        latest = st.session_state.workflow_runs[0]
        st.divider()
        st.markdown("### 최근 실행 결과")

        for step in latest["steps"]:
            with st.expander(
                f"STEP {step['step']} · {step['agent_name']} · {step['model']}",
                expanded=True,
            ):
                st.markdown("**Output**")
                st.markdown(step["output"])

                with st.expander("이 단계에 실제 전달된 Input 보기"):
                    st.text(step["input"])

                if step["rag_hits"]:
                    with st.expander(f"RAG 검색 근거 {len(step['rag_hits'])}개 보기"):
                        for hit in step["rag_hits"]:
                            st.caption(f"{hit['source']} · chunk {hit['chunk_id']} · similarity {hit['score']:.3f}")
                            st.write(hit["text"])
                            st.markdown("---")

        if latest.get("final_output") is not None:
            st.markdown("### ✅ Final Output")
            st.markdown(latest["final_output"])

            if has_self_intro_workflow:
                submission = extract_final_submission(latest["final_output"])
                st.markdown("#### 📋 복사용 최종 제출본")
                st.code(submission, language=None)
                st.caption(
                    f"현재 최종 제출본 글자 수: 공백 포함 {len(submission):,}자 · "
                    f"공백 제외 {len(''.join(submission.split())):,}자"
                )

    if len(st.session_state.workflow_runs) > 1:
        st.caption(f"현재 세션에 {len(st.session_state.workflow_runs)}개의 실행 기록이 있습니다.")
        if st.button("실행 기록 전체 삭제"):
            st.session_state.workflow_runs = []
            st.rerun()


st.divider()
st.caption(
    "Security note: API key는 이 앱 코드에 하드코딩하지 않고 사용자가 웹 UI에서 직접 입력합니다. "
    "Streamlit Community Cloud에 공개 배포할 경우에도 API key를 GitHub 저장소에 커밋하지 마세요."
)
