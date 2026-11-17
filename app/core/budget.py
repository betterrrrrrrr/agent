"""上下文预算纯计算。字符估算不是模型 tokenizer；留安全余量并在调用前复核。"""
from dataclasses import dataclass
import math

from app.config import settings


def lookup_window(model: str) -> tuple[int, bool]:
    name = model.rsplit('/', 1)[-1].lower()
    # 账户/通道可能缩小窗口，MODEL_CONTEXT_WINDOW 始终优先。
    for prefix, size in (("deepseek-v4", 1000000), ("gpt-4.1", 1047576),
                         ("gpt-4o", 128000), ("deepseek-v3", 65536)):
        if name.startswith(prefix):
            return size, True
    return 32768, False


def resolve_window() -> int:
    return settings.model_context_window or lookup_window(settings.chat_model)[0]


def evidence_tokens() -> int:
    return max(1, settings.rerank_top_k) * settings.evidence_per_hit_tokens


def summary_tokens() -> int:
    return math.ceil(settings.summary_segment_max_chars / settings.zh_chars_per_token) * settings.summary_inject_segments


def history_per_turn() -> int:
    return settings.max_user_input_tokens + settings.max_output_tokens


def turn_peak_tokens() -> int:
    # 下一次模型调用前，最多已有 steps-1 组 AI 输出+工具返回；最后输出另在 fixed 中留。
    return (settings.max_user_input_tokens + (settings.max_agent_steps - 1) * (
        settings.max_output_tokens + settings.max_tool_calls_per_step * settings.tool_result_max_tokens))


@dataclass(frozen=True)
class ContextBudget:
    window: int
    fixed: int
    turn_peak: int
    sliding: int
    layer1: int
    layer2: int
    healthy: bool


def compute() -> ContextBudget:
    window = resolve_window()
    fixed = (settings.system_prompt_tokens + evidence_tokens() + summary_tokens()
             + settings.max_output_tokens + settings.budget_safety_margin)
    peak = turn_peak_tokens()
    room = window - fixed - peak
    sliding = max(0, min(settings.context_budget_turns * history_per_turn(), room))
    if settings.context_window_max_tokens:
        sliding = min(sliding, settings.context_window_max_tokens)
    layer1 = int(sliding * settings.layer1_ratio)
    return ContextBudget(window, fixed, peak, sliding, layer1, sliding - layer1,
                         room > 0 and layer1 > 0)


def startup_check(logger) -> ContextBudget:
    b = compute()
    if not settings.model_context_window and not lookup_window(settings.chat_model)[1]:
        logger.warning('未知模型窗口，使用保守值 %s；请设置 MODEL_CONTEXT_WINDOW', b.window)
    logger.info('context_budget window=%s fixed=%s peak=%s sliding=%s layer1=%s layer2=%s',
                b.window, b.fixed, b.turn_peak, b.sliding, b.layer1, b.layer2)
    if not b.healthy:
        logger.error('上下文预算不足：无法同时保证目标历史和单轮上限，请调整窗口/输入/输出/步数')
    return b
