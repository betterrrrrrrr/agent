import asyncio
import pytest
from sqlalchemy import select
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import settings
from app.core import budget, memory, summarizer
from app.db import repository as repo
from app.db.models import ConversationSummary


async def segments(factory, cid):
    async with factory() as s:
        return list((await s.execute(select(ConversationSummary).where(
            ConversationSummary.conversation_id == cid).order_by(ConversationSummary.seq))).scalars())


async def test_segments_append_projection_and_reject_stale(db_session_factory, monkeypatch):
    monkeypatch.setattr(settings, 'summary_inject_segments', 1)
    cid = await repo.create_conversation('user')
    await repo.update_context_layers(cid, 8, 100, 0)
    assert await repo.append_summary_segment(cid, from_msg_id=1, upto_msg_id=4, content='第一段订单1001') == 1
    assert await repo.append_summary_segment(cid, from_msg_id=1, upto_msg_id=8, content='过期段') is None
    assert await repo.append_summary_segment(cid, from_msg_id=5, upto_msg_id=8, content='第二段') == 2
    conv = await repo.get_conversation(cid)
    assert conv.summary == '第二段' and conv.summary_upto_msg_id == 8
    assert [s.content for s in await segments(db_session_factory, cid)] == ['第一段订单1001', '第二段']


async def test_summary_never_consumes_layer1_and_failure_keeps_boundary(db_session_factory, monkeypatch):
    cid = await repo.create_conversation('user')
    for i in range(4):
        await repo.append_message(cid, 'user', f'问题{i}')
        await repo.append_message(cid, 'assistant', f'答复{i}')
    await repo.update_context_layers(cid, 4, 1000, 0)
    seen = []
    async def fake(old, text):
        seen.append(text)
        return '用户在第一二轮询问订单1001'
    monkeypatch.setattr(summarizer, 'summarize_dialog', fake)
    await summarizer.run_summary(cid)
    assert '问题1' in seen[0] and '问题2' not in seen[0]
    await repo.update_context_layers(cid, 6, 1000, 4)
    async def empty(*args):
        return ''
    monkeypatch.setattr(summarizer, 'summarize_dialog', empty)
    with pytest.raises(ValueError):
        await summarizer.run_summary(cid)
    assert (await repo.get_conversation(cid)).summary_upto_msg_id == 4
    assert len(await segments(db_session_factory, cid)) == 1


async def test_legacy_summary_preserved_without_rewrite(db_session_factory):
    cid = await repo.create_conversation('user')
    await repo.update_conversation_summary(cid, '旧事实订单1001', 4)
    await repo.update_context_layers(cid, 8, 1000, 4)
    await repo.append_summary_segment(cid, from_msg_id=5, upto_msg_id=8, content='新片段事实')
    assert [s.content for s in await segments(db_session_factory, cid)] == ['旧事实订单1001', '新片段事实']


async def test_more_than_twenty_turns_cascade_uses_checkpoint_tools(db_session_factory, monkeypatch):
    # 缩小历史旋钮制造压力，走真实仓储事务及任务调度，模型替身只负责可重复的段落输出。
    monkeypatch.setattr(settings, 'context_window_max_tokens', 700)
    monkeypatch.setattr(settings, 'layer1_ratio', 0.7)
    cid = await repo.create_conversation('user')
    messages, summaries = [], []
    async def fake(old, text):
        summaries.append((old, text))
        return '用户咨询订单1001物流及开票诉求' if '1001' in text else '用户继续咨询物流及发票'
    monkeypatch.setattr(summarizer, 'summarize_dialog', fake)
    for i in range(36):
        q = '订单1001多久到，要求开发票' if i == 0 else f'物流进度和发票问题{i}'
        mid = await repo.append_message(cid, 'user', q)
        await repo.append_message(cid, 'assistant', '物流回复' * 30)
        messages.extend([HumanMessage(q, id=f'db-{mid}'),
            AIMessage('', tool_calls=[{'id': f'c{i}', 'name': 'q', 'args': {}}]),
            ToolMessage('工具内容' * 200, tool_call_id=f'c{i}', name='q'), AIMessage('物流回复' * 30)])
        original = [m.model_dump() for m in messages]
        await summarizer.settle_layers(cid, messages)
        await summarizer.maybe_schedule_summary(cid)
        task = summarizer._running.get(cid)
        if task:
            await task
        conv = await repo.get_conversation(cid)
        win = memory.build_window(messages, conv.summary_upto_msg_id or 0,
            layer1_from_msg_id=conv.layer1_from_msg_id or 0)
        assert win[0].type == 'human'
        assert [m.model_dump() for m in messages] == original
    saved = await segments(db_session_factory, cid)
    assert len(saved) >= 2 and summaries
    assert '1001' in saved[0].content
    assert all(a.upto_msg_id + 1 == b.from_msg_id for a, b in zip(saved, saved[1:]))
    assert len(await repo.list_dialog_messages(cid)) == 72


async def test_many_messages_under_large_budget_do_not_trigger(db_session_factory, monkeypatch):
    cid = await repo.create_conversation('user')
    messages = []
    for i in range(25):
        mid = await repo.append_message(cid, 'user', '问题')
        await repo.append_message(cid, 'assistant', '答复')
        messages.extend([HumanMessage('问题', id=f'db-{mid}'), AIMessage('答复')])
    await summarizer.settle_layers(cid, messages)
    await summarizer.maybe_schedule_summary(cid)
    assert cid not in summarizer._running
    assert (await repo.get_conversation(cid)).layer1_from_msg_id == 0


async def test_long_single_turn_rejected_before_database(monkeypatch):
    from app.graph import runtime
    monkeypatch.setattr(settings, 'max_user_input_tokens', 5)
    with pytest.raises(ValueError, match='单轮输入'):
        await runtime.run_turn('user', '过长输入' * 10, None)


@pytest.mark.parametrize('streaming', [False, True])
async def test_runtime_reloads_two_boundaries_and_keeps_complete_state(db_session_factory, monkeypatch, streaming):
    from app.graph import runtime, nodes
    from app.graph.state import ConversationState
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    monkeypatch.setattr(settings, 'context_window_max_tokens', 350)
    async def reply(state):
        return {'messages': [AIMessage('查询结果：订单1001仍在运输，请耐心等待。' * 3)]}
    graph = StateGraph(ConversationState)
    graph.add_node('main_agent', reply)
    graph.add_node('log', nodes.log_node)
    graph.add_edge(START, 'main_agent')
    graph.add_edge('main_agent', 'log')
    graph.add_edge('log', END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(runtime, 'get_graph', lambda: compiled)
    async def fake_summary(old, text):
        return '用户咨询订单1001物流，客服已说明仍在运输。'
    monkeypatch.setattr(summarizer, 'summarize_dialog', fake_summary)
    cid = None
    for i in range(24):
        text = f'订单1001物流第{i}次查询，请核实进度。'
        if streaming:
            events = [e async for e in runtime.stream_turn('u', text, cid)]
            cid = events[-1]['conversation_id']
        else:
            cid = (await runtime.run_turn('u', text, cid))['conversation_id']
        task = summarizer._running.get(cid)
        if task:
            await task
    conv = await repo.get_conversation(cid)
    assert conv.summary_upto_msg_id > 0 and conv.layer1_from_msg_id >= conv.summary_upto_msg_id
    snap = await compiled.aget_state({'configurable': {'thread_id': str(cid)}})
    assert len(snap.values['messages']) == 48
    assert snap.values['layer1_from_msg_id'] > 0
    assert '1001' in snap.values['summary']
    assert len(await repo.list_dialog_messages(cid)) == 48


async def test_shutdown_cancels_pending_summary_without_advancing_boundary(db_session_factory, monkeypatch):
    cid = await repo.create_conversation('u')
    await repo.update_context_layers(cid, 2, 1000000, 0)
    gate = asyncio.Event()
    async def blocked(cid):
        await gate.wait()
    monkeypatch.setattr(summarizer, 'run_summary', blocked)
    await summarizer.maybe_schedule_summary(cid)
    task = summarizer._running[cid]
    await summarizer.close()
    assert task.cancelled() and not summarizer._running
    assert not (await repo.get_conversation(cid)).summary_upto_msg_id
