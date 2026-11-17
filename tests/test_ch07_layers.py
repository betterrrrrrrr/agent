import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import settings
from app.core import memory


def dialog(n=5):
    result = []
    for i in range(n):
        result.extend([HumanMessage('订单1001问题' + str(i), id=f'db-{i*2+1}'),
                       AIMessage('回答' * 100, id=f'ai-{i}')])
    return result


def test_three_layers_preserve_original_and_tool_links():
    msgs = dialog()
    msgs[3] = AIMessage('很长的回答' * 100, id='ai-tool',
                        tool_calls=[{'id': 'call-1', 'name': 'query_order', 'args': {}}])
    msgs.insert(4, ToolMessage('物流结果' * 500, tool_call_id='call-1', name='query_order'))
    before = [m.model_dump() for m in msgs]
    win = memory.build_window(msgs, 2, layer1_from_msg_id=6, max_tokens=100000)
    assert win[0].id == 'db-3'
    assert len(win[1].content) < len(msgs[3].content)
    assert win[1].tool_calls == msgs[3].tool_calls
    assert win[2].tool_call_id == 'call-1' and '结果从略' in win[2].content
    assert win[-1].content == msgs[-1].content
    assert [m.model_dump() for m in msgs] == before
    assert [m.model_dump() for m in win] == [m.model_dump() for m in memory.build_window(
        msgs, 2, layer1_from_msg_id=6, max_tokens=100000)]


def test_boundary_counts_tool_payload_and_keeps_latest_turn():
    msgs = dialog(3)
    msgs.insert(2, ToolMessage('大' * 5000, tool_call_id='c', name='query'))
    boundary = memory.compute_layer1_boundary(msgs, 0, 800)
    assert boundary >= 2
    assert boundary < 5


def test_dynamic_budget_and_unhealthy_window(monkeypatch):
    from app.core import budget
    monkeypatch.setattr(settings, 'model_context_window', 18000)
    monkeypatch.setattr(settings, 'max_output_tokens', 2000)
    monkeypatch.setattr(settings, 'max_user_input_tokens', 2000)
    monkeypatch.setattr(settings, 'max_agent_steps', 3)
    monkeypatch.setattr(settings, 'tool_result_max_tokens', 1200)
    monkeypatch.setattr(settings, 'rerank_top_k', 5)
    b = budget.compute()
    assert b.healthy and b.sliding > 0
    assert (b.fixed, b.turn_peak, b.sliding, b.layer1, b.layer2) == (6350, 8400, 3250, 2275, 975)
    assert b.fixed + b.turn_peak + b.sliding <= b.window
    assert b.layer1 + b.layer2 == b.sliding
    monkeypatch.setattr(settings, 'model_context_window', 2000)
    assert not budget.compute().healthy


def test_default_large_window_keeps_twenty_turns(monkeypatch):
    from app.core import budget
    monkeypatch.setattr(settings, 'model_context_window', 1000000)
    monkeypatch.setattr(settings, 'context_window_max_tokens', 0)
    assert memory.compute_layer1_boundary(dialog(20), 0, budget.compute().layer1) == 0


def test_chinese_token_conversion_has_one_direction():
    assert memory.chars_to_tokens(100) == __import__('math').ceil(100 / settings.zh_chars_per_token)


def test_current_tool_round_is_not_trimmed_in_half():
    msgs = dialog(2)
    msgs += [HumanMessage('最新', id='db-5'), AIMessage('', tool_calls=[
        {'id': 'c', 'name': 'q', 'args': {}}]), ToolMessage('结果' * 500, tool_call_id='c')]
    win = memory.build_window(msgs, 0, max_tokens=10)
    assert win[0].id == 'db-5'
    assert win[-2].tool_calls[0]['id'] == win[-1].tool_call_id


def test_invalid_configuration_and_manual_cap(monkeypatch):
    from app.config import Settings
    from app.core import budget
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(layer1_ratio=1.0)
    with pytest.raises(ValidationError):
        Settings(zh_chars_per_token=0)
    monkeypatch.setattr(settings, 'model_context_window', 1000000)
    monkeypatch.setattr(settings, 'context_window_max_tokens', 1234)
    assert budget.compute().sliding == 1234
    assert budget.lookup_window('deepseek-ai/DeepSeek-V4-Flash') == (1000000, True)


def test_api_input_limits_and_output_limit(monkeypatch):
    from app.schemas.chat import ChatRequest
    from app.schemas.agent import AgentRequest
    from app.core.llm import get_chat_model
    from pydantic import ValidationError
    monkeypatch.setattr(settings, 'max_user_input_tokens', 10)
    for schema in (ChatRequest, AgentRequest):
        with pytest.raises(ValidationError):
            schema(user_id='u', message='中' * 11)
    assert get_chat_model().max_tokens == settings.max_output_tokens


def test_tool_output_is_bounded(monkeypatch):
    monkeypatch.setattr(settings, 'zh_chars_per_token', 1.5)
    result = memory.limit_text('工具结果' * 100, 20)
    assert memory.chars_to_tokens(len(result)) <= 20


async def test_ticket_outside_step_limit_does_not_ask_for_confirmation(monkeypatch):
    from types import SimpleNamespace
    from app.graph import nodes
    monkeypatch.setattr(settings, 'max_tool_calls_per_step', 1)
    async def specs():
        return [SimpleNamespace(name='create_ticket')]
    async def execute(tc, *args, **kwargs):
        return SimpleNamespace(tool_message=ToolMessage('已查询', tool_call_id=tc['id'], name=tc['name']))
    monkeypatch.setattr(nodes.registry, 'get_all_specs', specs)
    monkeypatch.setattr(nodes.engine, 'execute_tool_call', execute)
    def should_not_interrupt(*args):
        pytest.fail('超出执行上限的写调用不应要求用户确认')
    monkeypatch.setattr(nodes, 'interrupt', should_not_interrupt)
    result = await nodes.agent_tools({'messages': [AIMessage('', tool_calls=[
        {'id': 'q', 'name': 'query_product', 'args': {}},
        {'id': 't', 'name': 'create_ticket', 'args': {'description': '投诉'}}])]})
    assert len(result['messages']) == 2
    assert result['messages'][1].tool_call_id == 't'
    assert '未执行' in result['messages'][1].content
