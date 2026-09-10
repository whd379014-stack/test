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


DEFAULT_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
]
EMBEDDING_MODEL = "text-embedding-3-small"


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
        "model": "gpt-5-mini",
        "custom_model": "",
        "system_prompt": "You are a helpful AI agent. Follow the user's instructions accurately.",
        "rag_enabled": False,
        "rag_top_k": 4,
        "rag_chunks": [],
        "rag_embeddings": None,
        "rag_files": [],
        "rag_fingerprint": "",
    }


def effective_model(agent: Dict[str, Any]) -> str:
    custom = agent.get("custom_model", "").strip()
    return custom if custom else agent.get("model", "gpt-5-mini")


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
                client.models.list()
                st.session_state.api_ok = True
                st.success("연결 성공")
            except Exception as e:
                st.session_state.api_ok = False
                st.error(f"연결 실패: {e}")
    with c2:
        if st.button("키 지우기", use_container_width=True):
            st.session_state.api_key = ""
            st.session_state.api_ok = False
            st.rerun()

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
        st.caption(f"현재 {len(st.session_state.agents)}개")

    with top_right:
        st.info("각 에이전트의 **System Prompt**와 모델을 지정한 뒤 먼저 개별 채팅으로 테스트할 수 있습니다. RAG는 에이전트별로 독립 설정됩니다.")

    if not st.session_state.agents:
        st.warning("에이전트가 없습니다. 먼저 '에이전트 추가'를 눌러 주세요.")

    for agent_id in list(st.session_state.agents.keys()):
        agent = st.session_state.agents[agent_id]
        with st.expander(f"🤖 {agent['name']}  ·  {effective_model(agent)}", expanded=True):
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
                    index=DEFAULT_MODELS.index(agent["model"]) if agent["model"] in DEFAULT_MODELS else 1,
                    key=f"model_{agent_id}",
                )
                agent["custom_model"] = st.text_input(
                    "Custom model ID (선택)",
                    value=agent.get("custom_model", ""),
                    placeholder="예: gpt-5.1-mini",
                    key=f"custom_model_{agent_id}",
                    help="입력하면 위 기본 모델보다 우선합니다.",
                )
                if st.button("에이전트 삭제", key=f"delete_{agent_id}", use_container_width=True):
                    st.session_state.agents.pop(agent_id, None)
                    st.session_state.chat_histories.pop(agent_id, None)
                    remove_from_workflow(agent_id)
                    st.rerun()

            st.markdown("#### 선택적 RAG")
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

    if st.session_state.workflow:
        names = [st.session_state.agents[aid]["name"] for aid in st.session_state.workflow if aid in st.session_state.agents]
        st.code("  →  ".join(names), language=None)
    else:
        st.warning("먼저 Linear Workflow를 구성해 주세요.")

    user_prompt = st.text_area(
        "최초 User Prompt",
        height=150,
        placeholder="예: 이 고객 VOC 데이터를 분석해서 핵심 불만 유형과 개선 액션을 도출해줘.",
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
