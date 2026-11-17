# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import math
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from app.config import settings
from app.core import budget


def chars_to_tokens(n: int) -> int:
    return math.ceil(n / settings.zh_chars_per_token)


def count_tokens(messages, **kwargs) -> int:
    return count_tokens_approximately(messages, chars_per_token=settings.zh_chars_per_token, **kwargs)


def limit_text(text: str, tokens: int) -> str:
    """文本上限含省略标记；所有字符换算均使用同一个除数。"""
    n = max(0, int(tokens * settings.zh_chars_per_token))
    if len(text) <= n:
        return text
    marker = '…(截断)'
    return text[:max(0, n-len(marker))] + marker[:n]


def validate_user_input(text: str) -> str:
    if chars_to_tokens(len(text)) > settings.max_user_input_tokens:
        raise ValueError(f'单轮输入超过 {settings.max_user_input_tokens} token，请缩短后重试')
    return text


def window_budget() -> int:
    return budget.compute().sliding


class SessionStore:
    """内存会话存储,session_id -> 消息列表。本章不做持久化。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, *messages: BaseMessage) -> None:
        self._sessions.setdefault(session_id, []).extend(messages)


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    return trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens,
        max_tokens=max_tokens,
        start_on="human",
        allow_partial=False,
    )


# ---- ch07 两个锚点切窗 + 三层渲染 + 摘要注入 ----

_DB_ID_PREFIX = "db-"


def _db_msg_id(m: BaseMessage) -> int | None:
    """从消息 id 解出 MySQL 消息 id(入口约定 HumanMessage.id=f"db-{msg_id}");非锚点返回 None。"""
    mid = getattr(m, "id", None)
    if isinstance(mid, str) and mid.startswith(_DB_ID_PREFIX):
        try:
            return int(mid[len(_DB_ID_PREFIX):])
        except ValueError:
            return None
    return None


def _index_after(messages, msg_id: int) -> int:
    if not msg_id:
        return 0
    for i, m in enumerate(messages):
        mid = _db_msg_id(m)
        if isinstance(m, HumanMessage) and mid is not None and mid > msg_id:
            return i
    return 0  # 旧 checkpoint 没有 DB 坐标时退化为 token 裁剪


def compress_reply(text: str, keep_chars: int | None = None) -> str:
    n = settings.layer2_reply_keep_chars if keep_chars is None else keep_chars
    return text if len(text) <= n else text[:n] + '…(略)'


def compress_tool_result(name: str, content: str) -> str:
    if chars_to_tokens(len(content)) <= settings.layer2_tool_keep_tokens:
        return content
    return f'(已调用 {name or "工具"},结果从略)'


def to_layer2(messages) -> list[BaseMessage]:
    result = []
    for m in messages:
        if isinstance(m, (AIMessage, ToolMessage)):
            text = m.content if isinstance(m.content, str) else m.text
            content = (compress_reply(text) if isinstance(m, AIMessage)
                       else compress_tool_result(m.name or '', text))
            result.append(m.model_copy(update={'content': content}))
        else:
            result.append(m)
    return result


def compute_layer1_boundary(messages, previous: int, max_tokens: int) -> int:
    """按完整用户轮降级，最新轮始终留在层1；包括 AI 工具调用和返回的用量。"""
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage) and _db_msg_id(m) is not None]
    if not starts:
        return previous
    keep = starts[-1]
    for start in reversed(starts[:-1]):
        mid = _db_msg_id(messages[start])
        if mid <= previous or count_tokens(messages[start:]) > max_tokens:
            break
        keep = start
    first = _db_msg_id(messages[keep])
    return max(previous, first - 1) if keep > starts[0] else previous


def layer2_tokens(messages, upto: int, layer1_from: int) -> int:
    if layer1_from <= upto:
        return 0
    start = _index_after(messages, upto)
    stop = _index_after(messages, layer1_from)
    return count_tokens(to_layer2(messages[start:stop])) if stop > start else 0


def build_window(messages: list[BaseMessage], summary_upto_msg_id: int,
                 layer1_from_msg_id: int = 0, max_tokens: int | None = None) -> list[BaseMessage]:
    """临时渲染三层，State 原文不动。token 兜底按完整用户轮切割，保留当前工具环。"""
    window = messages[_index_after(messages, summary_upto_msg_id):]
    split = _index_after(window, layer1_from_msg_id)
    if split:
        window = to_layer2(window[:split]) + window[split:]
    cap = window_budget() if max_tokens is None else max_tokens
    # trim_messages 的尾切可能只留下部分工具返回，改为在完整用户轮边界裁剪。
    starts = [i for i, m in enumerate(window) if isinstance(m, HumanMessage)]
    for i in starts:
        candidate = window[i:]
        if count_tokens(candidate) <= cap:
            return trim_history(candidate, cap) or candidate
    return window[starts[-1]:] if starts else window


def summary_line(summary: str | None) -> str:
    """coref/意图的历史文本前缀:单行摘要;无摘要空串。"""
    return f"(早前对话摘要:{summary})" if summary else ""


def summary_system(summary: str | None) -> SystemMessage | None:
    """兼容辅助函数：调用者只取 content 放入本轮材料，不作为 system 发出。"""
    if not summary:
        return None
    return SystemMessage(f"## 早前对话摘要(更早轮次已压缩,其中事实可信)\n{summary}")
