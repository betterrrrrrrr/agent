# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch07 后台分段摘要:层2超预算触发,asyncio 后台跑,不阻塞当轮回复。
失败只 log 不重试——触发条件仍满足,下一轮自然重触发;summary 两字段只在成功后原子更新。"""
import asyncio
import logging
import time

from pydantic import BaseModel, Field

from app.config import settings
from app.core import budget, llm, memory
from app.core.prompts import SUMMARY_PROMPT
from app.db import repository

logger = logging.getLogger(__name__)

_running: dict[int, asyncio.Task] = {}   # cid -> 在跑任务(防抖:在跑不重复起)


class _Summary(BaseModel):
    summary: str = Field(min_length=1, description="仅本段对话的事实与诉求")


def compute_boundary(messages, keep_turns: int) -> int | None:
    """旧教学版兼容函数（生产三层路径不调用）。
    新摘要边界 = 倒数第 keep_turns 轮用户消息的前一条消息 id。
    不足 keep_turns 轮 / 边界落在会话开头 → None(本次放弃)。
    messages:按 id 升序的 user/assistant 行(需 .id/.role)。"""
    user_idx = [i for i, m in enumerate(messages) if m.role == "user"]
    if len(user_idx) <= keep_turns:
        return None
    cut = user_idx[-keep_turns]
    if cut == 0:
        return None
    return messages[cut - 1].id


async def summarize_dialog(old_summary: str, dialog: str) -> str:
    """旧摘要只作消歧背景，新段仅压缩一次，不合并重写旧事实。"""
    model = llm.structured(_Summary, model=settings.summary_model or None)
    r: _Summary = await (SUMMARY_PROMPT | model).ainvoke(
        {"old_summary": old_summary or "(无)", "dialog": dialog,
         "max_chars": settings.summary_segment_max_chars})
    result = (r.summary or "").strip()
    if not result or len(result) > settings.summary_segment_max_chars:
        raise ValueError('摘要为空或超过段落字数上限，放弃推进边界')
    return result


async def run_summary(conversation_id: int) -> None:
    """任务体:读消息 → 算边界 → 压缩 → 原子更新两字段。异常往外抛,由 done 回调统一 log。"""
    t0 = time.monotonic()
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        return
    msgs = await repository.list_dialog_messages(conversation_id)
    boundary = conv.layer1_from_msg_id or 0
    old_upto = conv.summary_upto_msg_id or 0
    if boundary <= old_upto:
        logger.info("summary skip conv=%s boundary=%s upto=%s",
                    conversation_id, boundary, old_upto)
        return
    seg = [m for m in msgs if old_upto < m.id <= boundary]
    dialog = "\n".join(
        f"{'用户' if m.role == 'user' else '客服'}:{m.content}" for m in seg if m.content)
    logger.info("summary start conv=%s msgs=%s upto %s->%s",
                conversation_id, len(seg), old_upto, boundary)
    summary = await summarize_dialog(conv.summary or "", dialog)
    if not summary:
        raise ValueError("摘要为空,放弃更新")
    if len(summary) > settings.summary_segment_max_chars:
        raise ValueError('摘要超过字数上限，放弃更新')
    seq = await repository.append_summary_segment(conversation_id, from_msg_id=old_upto + 1,
                upto_msg_id=boundary, content=summary)
    if seq is None:
        logger.info('summary skip conv=%s 任务边界已过期 upto=%s', conversation_id, boundary)
        return
    logger.info("summary done conv=%s 第%s段 upto=%s len=%s cost=%.0fms",
                conversation_id, seq, boundary, len(summary), (time.monotonic() - t0) * 1000)


async def settle_layers(conversation_id: int, messages: list) -> None:
    """轮末从完整 State 计算降级/用量，包括不落 MySQL 的工具结果。异常不破坏回复。"""
    try:
        conv = await repository.get_conversation(conversation_id)
        if conv is None:
            return
        upto = conv.summary_upto_msg_id or 0
        previous = max(conv.layer1_from_msg_id or 0, upto)
        boundary = memory.compute_layer1_boundary(messages, previous, budget.compute().layer1)
        tokens = memory.layer2_tokens(messages, upto, boundary)
        saved = await repository.update_context_layers(conversation_id, boundary, tokens, upto)
        if saved and boundary > previous:
            logger.info('层1 降级 conv=%s %s→%s 层2≈%s tokens', conversation_id, previous, boundary, tokens)
    except Exception:
        logger.exception('context layers settle failed conv=%s', conversation_id)


async def _should_summarize(conversation_id: int, conv) -> bool:
    if (conv.layer1_from_msg_id or 0) <= (conv.summary_upto_msg_id or 0):
        return False
    return conv.layer2_tokens > budget.compute().layer2


async def maybe_schedule_summary(conversation_id: int) -> None:
    """轮结束后调用:层2超过预算且无在跑任务 → create_task 后台跑,不 await。
    本函数在用户回复路径上,任何异常只 log 不外抛(触发条件仍在,下一轮重来)。"""
    try:
        if conversation_id in _running:
            return
        conv = await repository.get_conversation(conversation_id)
        if conv is None:
            return
        if not await _should_summarize(conversation_id, conv):
            return
        # 上面两个 await 期间可能有并发触发已入表(TOCTOU),入表前再查一次防双起
        if conversation_id in _running:
            return
        logger.info("summary trigger conv=%s 层2约%s token > 预算%s", conversation_id,
                    conv.layer2_tokens, budget.compute().layer2)
        task = asyncio.create_task(run_summary(conversation_id))
        _running[conversation_id] = task

        def _done(t: asyncio.Task) -> None:
            if _running.get(conversation_id) is t:   # 只清自己,防误清后来者
                _running.pop(conversation_id, None)
            if not t.cancelled() and t.exception() is not None:
                logger.error("summary failed conv=%s", conversation_id, exc_info=t.exception())

        task.add_done_callback(_done)
    except Exception:
        logger.exception("summary trigger check failed conv=%s", conversation_id)


async def close() -> None:
    """应用停止时收回后台任务；未提交段落保持旧边界，下次请求可重新触发。"""
    tasks = list(_running.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _running.clear()
