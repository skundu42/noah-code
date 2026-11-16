"""Curated catalog of mainstream coding models (2025-2026) and OpenRouter routes.

Supplements LiteLLM's static backup JSON so users can pick any recent coding
model or any OpenAI / Anthropic model on OpenRouter directly from the TUI
picker without typing raw model identifiers, even offline or before API-key
entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CuratedModelSpec:
    """Metadata for a first-class selectable model route."""

    provider: str  # Noah provider key (e.g. "openrouter", "openai", "anthropic")
    model_id: str  # Provider-relative ID shown in picker (e.g. "anthropic/claude-sonnet-5.5")
    context_tokens: int
    tag: str  # Short capability label shown in the picker description
    rank: int  # Lower sorts higher in the picker; < 100 = featured coding tier
    quick_pick: bool = False  # Included in cross-provider "Popular coding models" picker


def _entry(
    litellm_provider: str,
    context_tokens: int,
    tag: str,
    rank: int,
) -> dict[str, Any]:
    return {
        "litellm_provider": litellm_provider,
        "mode": "chat",
        "max_input_tokens": context_tokens,
        "supports_function_calling": True,
        "_curated_tag": tag,
        "_curated_rank": rank,
    }


# ---------------------------------------------------------------------------
# 1. OpenRouter: Routers + ALL Anthropic + ALL OpenAI + Last-12-Months Coding
# ---------------------------------------------------------------------------

_OPENROUTER_SPECS: tuple[tuple[str, int, str, int, bool], ...] = (
    # Featured Coding & Agentic Flagships (Last 12 Months) — Ranks 1..65
    ("openrouter/auto-code", 2_000_000, "★ Coding router · auto-selects top coding model", 1, True),
    ("anthropic/claude-sonnet-5.5", 200_000, "★ Coding flagship · Anthropic Claude Sonnet 5.5", 2, True),
    ("anthropic/claude-opus-5.5", 200_000, "★ Coding & architecture · Anthropic Claude Opus 5.5", 3, True),
    ("openai/gpt-5.3-codex", 400_000, "★ Agentic coding · OpenAI GPT-5.3 Codex", 4, True),
    ("openai/gpt-6.1-terra-pro", 1_050_000, "★ Frontier reasoning & code · OpenAI GPT-6.1 Terra Pro", 5, True),
    ("openai/gpt-6.1-terra", 1_050_000, "★ Frontier coding · OpenAI GPT-6.1 Terra", 6, True),
    ("anthropic/claude-sonnet-5", 200_000, "★ Coding workhorse · Anthropic Claude Sonnet 5", 7, True),
    ("anthropic/claude-opus-5", 200_000, "★ Deep reasoning & code · Anthropic Claude Opus 5", 8, True),
    ("anthropic/claude-sonnet-4.6", 200_000, "★ Coding workhorse · Anthropic Claude Sonnet 4.6", 9, True),
    ("anthropic/claude-opus-4.8", 200_000, "★ Deep refactoring · Anthropic Claude Opus 4.8", 10, True),
    ("openai/gpt-5.5", 1_050_000, "★ Frontier coding · OpenAI GPT-5.5", 11, True),
    ("openai/gpt-5.4", 1_050_000, "★ Coding & tool use · OpenAI GPT-5.4", 12, True),
    ("openai/gpt-5.2-codex", 400_000, "★ Agentic coding · OpenAI GPT-5.2 Codex", 13, True),
    ("openai/gpt-5.1-codex-max", 400_000, "★ Long-horizon coding · OpenAI GPT-5.1 Codex Max", 14, True),
    ("qwen/qwen3-coder-next", 262_144, "★ Open-weight coding · Qwen3 Coder Next", 15, True),
    ("qwen/qwen4-max", 1_000_000, "★ Frontier coding · Alibaba Qwen4 Max", 16, True),
    ("deepseek/deepseek-v4.1-flash", 1_048_576, "★ Fast high-value coding · DeepSeek V4.1 Flash", 17, True),
    ("deepseek/deepseek-v4-pro", 1_048_576, "★ Frontier reasoning & code · DeepSeek V4 Pro", 18, True),
    ("google/gemini-4-hydrogen", 1_048_576, "★ 1M-ctx coding · Google Gemini 4 Hydrogen", 19, True),
    ("google/gemini-3.8-flash", 1_048_576, "★ Fast 1M-ctx coding · Google Gemini 3.8 Flash", 20, True),
    ("google/gemini-3.1-pro-preview", 1_048_576, "★ 1M-ctx architecture · Google Gemini 3.1 Pro", 21, True),
    ("x-ai/grok-build-0.1", 256_000, "★ Agentic engineering · xAI Grok Build 0.1", 22, True),
    ("x-ai/grok-4.7", 500_000, "★ Frontier coding · xAI Grok 4.7", 23, True),
    ("x-ai/grok-code-fast-1", 256_000, "★ Low-latency coding · xAI Grok Code Fast 1", 24, True),
    ("mistralai/devstral-2512", 262_144, "★ SWE agent model · Mistral Devstral 2512", 25, True),
    ("mistralai/codestral-2508", 256_000, "★ Code generation · Mistral Codestral 2508", 26, True),
    ("mistralai/mistral-large-4-0", 1_048_576, "★ Frontier 1M-ctx · Mistral Large 4.0", 27, True),
    ("moonshotai/kimi-k2.7-code", 262_144, "★ Specialized coding · Moonshot Kimi K2.7 Code", 28, True),
    ("moonshotai/kimi-k3.1-pro", 1_048_576, "★ 1M-ctx agentic coding · Moonshot Kimi K3.1 Pro", 29, True),
    ("z-ai/glm-5.3-prime", 1_000_000, "★ Long-horizon coding · Z.ai GLM-5.3 Prime", 30, True),
    ("z-ai/glm-5.3", 1_048_576, "★ Agentic coding · Z.ai GLM-5.3", 31, True),
    ("z-ai/glm-5.2", 1_048_576, "★ SWE-bench coding · Z.ai GLM-5.2", 32, True),
    ("minimax/minimax-m3.1-flash", 1_048_576, "★ Fast 1M-ctx coding · MiniMax M3.1 Flash", 33, True),
    ("minimax/minimax-m3", 1_048_576, "★ Agentic coding · MiniMax M3", 34, True),
    ("openrouter/pareto-code", 2_000_000, "★ Cost/quality Pareto coding router", 35, False),
    ("openrouter/auto", 2_000_000, "OpenRouter dynamic model router", 36, False),
    ("openrouter/free", 200_000, "OpenRouter free-tier model router", 37, False),
    # Complete Anthropic Lineup on OpenRouter — Ranks 70..95
    ("anthropic/claude-haiku-5.5", 200_000, "Anthropic Claude Haiku 5.5 · fast coding", 70, False),
    ("anthropic/claude-fable-5.2", 200_000, "Anthropic Claude Fable 5.2", 71, False),
    ("anthropic/claude-fable-5.1", 200_000, "Anthropic Claude Fable 5.1", 72, False),
    ("anthropic/claude-fable-5", 200_000, "Anthropic Claude Fable 5", 73, False),
    ("anthropic/claude-opus-4.7", 200_000, "Anthropic Claude Opus 4.7", 74, False),
    ("anthropic/claude-opus-4.6", 200_000, "Anthropic Claude Opus 4.6", 75, False),
    ("anthropic/claude-opus-4.5", 200_000, "Anthropic Claude Opus 4.5", 76, False),
    ("anthropic/claude-opus-4.1", 200_000, "Anthropic Claude Opus 4.1", 77, False),
    ("anthropic/claude-opus-4", 200_000, "Anthropic Claude Opus 4", 78, False),
    ("anthropic/claude-sonnet-4.5", 200_000, "Anthropic Claude Sonnet 4.5", 79, False),
    ("anthropic/claude-sonnet-4", 200_000, "Anthropic Claude Sonnet 4", 80, False),
    ("anthropic/claude-haiku-4.5", 200_000, "Anthropic Claude Haiku 4.5 · fast & compact", 81, False),
    ("anthropic/claude-3.7-sonnet", 200_000, "Anthropic Claude 3.7 Sonnet", 82, False),
    ("anthropic/claude-3.7-sonnet:thinking", 200_000, "Anthropic Claude 3.7 Sonnet (thinking)", 83, False),
    ("anthropic/claude-3.5-sonnet", 200_000, "Anthropic Claude 3.5 Sonnet", 84, False),
    ("anthropic/claude-3.5-haiku", 200_000, "Anthropic Claude 3.5 Haiku", 85, False),
    ("anthropic/claude-3-opus", 200_000, "Anthropic Claude 3 Opus", 86, False),
    ("anthropic/claude-3-haiku", 200_000, "Anthropic Claude 3 Haiku", 87, False),
    # Complete OpenAI Lineup on OpenRouter — Ranks 100..155
    ("openai/gpt-6.1-sol-pro", 1_050_000, "OpenAI GPT-6.1 Sol Pro", 100, False),
    ("openai/gpt-6.1-sol", 1_050_000, "OpenAI GPT-6.1 Sol", 101, False),
    ("openai/gpt-6-astra-pro", 1_050_000, "OpenAI GPT-6 Astra Pro", 102, False),
    ("openai/gpt-6-astra", 1_050_000, "OpenAI GPT-6 Astra", 103, False),
    ("openai/gpt-6-sol-pro", 1_050_000, "OpenAI GPT-6 Sol Pro", 104, False),
    ("openai/gpt-6-sol", 1_050_000, "OpenAI GPT-6 Sol", 105, False),
    ("openai/gpt-6-luna-pro", 1_050_000, "OpenAI GPT-6 Luna Pro", 106, False),
    ("openai/gpt-6-luna", 1_050_000, "OpenAI GPT-6 Luna", 107, False),
    ("openai/gpt-5.6-terra-pro", 1_050_000, "OpenAI GPT-5.6 Terra Pro", 108, False),
    ("openai/gpt-5.6-terra", 1_050_000, "OpenAI GPT-5.6 Terra", 109, False),
    ("openai/gpt-5.6-sol-pro", 1_050_000, "OpenAI GPT-5.6 Sol Pro", 110, False),
    ("openai/gpt-5.6-sol", 1_050_000, "OpenAI GPT-5.6 Sol", 111, False),
    ("openai/gpt-5.6-luna-pro", 1_050_000, "OpenAI GPT-5.6 Luna Pro", 112, False),
    ("openai/gpt-5.6-luna", 1_050_000, "OpenAI GPT-5.6 Luna", 113, False),
    ("openai/gpt-5.5-pro", 1_050_000, "OpenAI GPT-5.5 Pro", 114, False),
    ("openai/gpt-5.4-pro", 1_050_000, "OpenAI GPT-5.4 Pro", 115, False),
    ("openai/gpt-5.4-mini", 400_000, "OpenAI GPT-5.4 Mini · fast coding", 116, False),
    ("openai/gpt-5.4-nano", 400_000, "OpenAI GPT-5.4 Nano · lightweight", 117, False),
    ("openai/gpt-5.2-pro", 400_000, "OpenAI GPT-5.2 Pro", 118, False),
    ("openai/gpt-5.2", 400_000, "OpenAI GPT-5.2", 119, False),
    ("openai/gpt-5.2-chat", 400_000, "OpenAI GPT-5.2 Chat", 120, False),
    ("openai/gpt-5.1-codex", 400_000, "OpenAI GPT-5.1 Codex", 121, False),
    ("openai/gpt-5.1-codex-mini", 400_000, "OpenAI GPT-5.1 Codex Mini", 122, False),
    ("openai/gpt-5.1", 400_000, "OpenAI GPT-5.1", 123, False),
    ("openai/gpt-5-codex", 400_000, "OpenAI GPT-5 Codex", 124, False),
    ("openai/gpt-5-pro", 400_000, "OpenAI GPT-5 Pro", 125, False),
    ("openai/gpt-5", 400_000, "OpenAI GPT-5", 126, False),
    ("openai/gpt-5-mini", 400_000, "OpenAI GPT-5 Mini", 127, False),
    ("openai/gpt-5-nano", 400_000, "OpenAI GPT-5 Nano", 128, False),
    ("openai/gpt-5-chat", 400_000, "OpenAI GPT-5 Chat", 129, False),
    ("openai/o4-mini", 200_000, "OpenAI o4-mini reasoning", 130, False),
    ("openai/o4-mini-high", 200_000, "OpenAI o4-mini (high reasoning)", 131, False),
    ("openai/o3-pro", 200_000, "OpenAI o3-pro reasoning", 132, False),
    ("openai/o3", 200_000, "OpenAI o3 reasoning", 133, False),
    ("openai/o3-mini-high", 200_000, "OpenAI o3-mini (high reasoning)", 134, False),
    ("openai/o3-mini", 200_000, "OpenAI o3-mini reasoning", 135, False),
    ("openai/o1", 200_000, "OpenAI o1 reasoning", 136, False),
    ("openai/gpt-4.1", 1_047_576, "OpenAI GPT-4.1 · 1M context", 137, False),
    ("openai/gpt-4.1-mini", 1_047_576, "OpenAI GPT-4.1 Mini · 1M context", 138, False),
    ("openai/gpt-4.1-nano", 1_047_576, "OpenAI GPT-4.1 Nano · 1M context", 139, False),
    ("openai/gpt-4o", 128_000, "OpenAI GPT-4o", 140, False),
    ("openai/gpt-4o-2024-11-20", 128_000, "OpenAI GPT-4o (2024-11-20)", 141, False),
    ("openai/gpt-4o-2024-08-06", 128_000, "OpenAI GPT-4o (2024-08-06)", 142, False),
    ("openai/gpt-4o-2024-05-13", 128_000, "OpenAI GPT-4o (2024-05-13)", 143, False),
    ("openai/gpt-4o-mini", 128_000, "OpenAI GPT-4o Mini", 144, False),
    ("openai/gpt-4o-mini-2024-07-18", 128_000, "OpenAI GPT-4o Mini (2024-07-18)", 145, False),
    ("openai/gpt-chat-latest", 128_000, "OpenAI ChatGPT latest", 146, False),
    ("openai/gpt-oss-120b", 131_072, "OpenAI gpt-oss-120b open-weight", 147, False),
    ("openai/gpt-oss-20b", 131_072, "OpenAI gpt-oss-20b open-weight", 148, False),
    ("openai/gpt-4", 8_192, "OpenAI GPT-4", 149, False),
    ("openai/gpt-3.5-turbo", 16_385, "OpenAI GPT-3.5 Turbo", 150, False),
    # Additional Mainstream Coding Models Released in Last 12 Months on OpenRouter — Ranks 160..240
    ("qwen/qwen3-coder-flash", 1_000_000, "Qwen3 Coder Flash · 1M-ctx fast coding", 160, False),
    ("qwen/qwen3-coder", 262_144, "Qwen3 Coder 480B A35B · coding specialist", 161, False),
    ("qwen/qwen3-coder-30b-a3b-instruct", 262_144, "Qwen3 Coder 30B A3B · fast MoE coding", 162, False),
    ("qwen/qwen3.8-max-prime", 1_000_000, "Qwen3.8 Max Prime · 1M context", 163, False),
    ("qwen/qwen3.8-max-0902", 1_000_000, "Qwen3.8 Max (0902) · 1M context", 164, False),
    ("qwen/qwen3.8-2.4t-a95b", 1_048_576, "Qwen3.8 2.4T A95B MoE", 165, False),
    ("qwen/qwen3.8-flash", 1_000_000, "Qwen3.8 Flash · 1M context", 166, False),
    ("qwen/qwen3.8-27b", 1_024_000, "Qwen3.8 27B · 1M context", 167, False),
    ("qwen/qwen3.7-max", 1_000_000, "Qwen3.7 Max · 1M context", 168, False),
    ("qwen/qwen3.7-plus", 1_000_000, "Qwen3.7 Plus · 1M context", 169, False),
    ("qwen/qwen3.7-flash", 1_000_000, "Qwen3.7 Flash · 1M context", 170, False),
    ("qwen/qwen3.6-plus", 1_000_000, "Qwen3.6 Plus · 1M context", 171, False),
    ("qwen/qwen3.6-flash", 1_000_000, "Qwen3.6 Flash · 1M context", 172, False),
    ("qwen/qwen3.6-35b-a3b", 262_144, "Qwen3.6 35B A3B MoE", 173, False),
    ("qwen/qwen3.6-27b", 262_144, "Qwen3.6 27B", 174, False),
    ("qwen/qwen3.5-plus-20260420", 1_000_000, "Qwen3.5 Plus · 1M context", 175, False),
    ("qwen/qwen3.5-397b-a17b", 262_144, "Qwen3.5 397B A17B MoE", 176, False),
    ("qwen/qwen3.5-122b-a10b", 262_144, "Qwen3.5 122B A10B MoE", 177, False),
    ("deepseek/deepseek-v4.1-flash-1017", 1_048_576, "DeepSeek V4.1 Flash (1017)", 178, False),
    ("deepseek/deepseek-v4-pro-0813", 1_048_576, "DeepSeek V4 Pro (0813)", 179, False),
    ("deepseek/deepseek-v4-flash", 1_048_576, "DeepSeek V4 Flash · 1M context", 180, False),
    ("deepseek/deepseek-v4-flash-0731", 1_048_576, "DeepSeek V4 Flash (0731)", 181, False),
    ("deepseek/deepseek-v3.2", 163_840, "DeepSeek V3.2 · coding & reasoning", 182, False),
    ("deepseek/deepseek-r1-0528", 163_840, "DeepSeek R1 (0528) reasoning", 183, False),
    ("deepseek/deepseek-r1", 163_840, "DeepSeek R1 reasoning", 184, False),
    ("deepseek/deepseek-chat-v3-0324", 163_840, "DeepSeek V3 (0324)", 185, False),
    ("google/gemini-4-carbon", 1_048_576, "Google Gemini 4 Carbon · 1M context", 186, False),
    ("google/gemini-4-argon", 1_048_576, "Google Gemini 4 Argon · 1M context", 187, False),
    ("google/gemini-3.7-flash", 1_048_576, "Google Gemini 3.7 Flash · 1M context", 188, False),
    ("google/gemini-3.6-flash", 1_048_576, "Google Gemini 3.6 Flash · 1M context", 189, False),
    ("google/gemini-3.5-flash", 1_048_576, "Google Gemini 3.5 Flash · 1M context", 190, False),
    ("google/gemini-3.5-flash-lite", 1_048_576, "Google Gemini 3.5 Flash Lite", 191, False),
    ("google/gemini-3.1-pro-preview-customtools", 1_048_576, "Google Gemini 3.1 Pro (custom tools)", 192, False),
    ("google/gemini-3.1-flash-lite", 1_048_576, "Google Gemini 3.1 Flash Lite", 193, False),
    ("google/gemini-3-flash-preview", 1_048_576, "Google Gemini 3 Flash", 194, False),
    ("google/gemini-2.5-pro", 1_048_576, "Google Gemini 2.5 Pro · 1M context", 195, False),
    ("google/gemini-2.5-flash", 1_048_576, "Google Gemini 2.5 Flash · 1M context", 196, False),
    ("google/gemma-4-31b-it", 262_144, "Google Gemma 4 31B Instruct", 197, False),
    ("google/gemma-4-26b-a4b-it", 262_144, "Google Gemma 4 26B A4B MoE", 198, False),
    ("x-ai/grok-4.6", 500_000, "xAI Grok 4.6 · 500k context", 199, False),
    ("x-ai/grok-4.5", 500_000, "xAI Grok 4.5 · 500k context", 200, False),
    ("x-ai/grok-4.3", 1_000_000, "xAI Grok 4.3 · 1M context", 201, False),
    ("x-ai/grok-4.20", 2_000_000, "xAI Grok 4.20 · 2M context", 202, False),
    ("x-ai/grok-4", 256_000, "xAI Grok 4", 203, False),
    ("mistralai/mistral-medium-3-5", 262_144, "Mistral Medium 3.5", 204, False),
    ("mistralai/mistral-small-2603", 262_144, "Mistral Small (2603)", 205, False),
    ("mistralai/ministral-14b-2512", 262_144, "Mistral Ministral 14B (2512)", 206, False),
    ("moonshotai/kimi-k3", 1_048_576, "Moonshot Kimi K3 · 1M context", 207, False),
    ("moonshotai/kimi-k2.6", 262_144, "Moonshot Kimi K2.6 · coding & agents", 208, False),
    ("moonshotai/kimi-k2.5", 262_144, "Moonshot Kimi K2.5 · coding & vision", 209, False),
    ("z-ai/glm-5.3-flashx", 1_048_576, "Z.ai GLM-5.3 FlashX · fast 1M context", 210, False),
    ("z-ai/glm-5.3-flash", 1_048_576, "Z.ai GLM-5.3 Flash · 1M context", 211, False),
    ("z-ai/glm-5.1", 204_800, "Z.ai GLM-5.1 · agentic coding", 212, False),
    ("z-ai/glm-5", 204_800, "Z.ai GLM-5 · agentic coding", 213, False),
    ("z-ai/glm-5-turbo", 202_752, "Z.ai GLM-5 Turbo", 214, False),
    ("z-ai/glm-4.7", 204_800, "Z.ai GLM-4.7 · coding", 215, False),
    ("z-ai/glm-4.7-flash", 200_000, "Z.ai GLM-4.7 Flash", 216, False),
    ("minimax/minimax-m2.7", 196_608, "MiniMax M2.7 · SWE coding", 217, False),
    ("minimax/minimax-m2.5", 198_000, "MiniMax M2.5 · SWE coding", 218, False),
    ("cohere/north-mini-code:free", 256_000, "Cohere North Mini Code (free tier)", 219, False),
    ("cohere/command-a-plus", 192_000, "Cohere Command A Plus", 220, False),
    ("meta-llama/llama-4-maverick", 1_048_576, "Meta Llama 4 Maverick", 221, False),
    ("meta-llama/llama-4-scout", 1_048_576, "Meta Llama 4 Scout", 222, False),
)


# ---------------------------------------------------------------------------
# 2. Direct Provider Curated Lineups (Last 12 Months Coding + Flagship Models)
# ---------------------------------------------------------------------------

_DIRECT_PROVIDER_SPECS: tuple[CuratedModelSpec, ...] = (
    # Anthropic Direct
    CuratedModelSpec("anthropic", "claude-sonnet-5-5", 200_000, "★ Coding flagship · Claude Sonnet 5.5", 1, True),
    CuratedModelSpec("anthropic", "claude-opus-5-5", 200_000, "★ Architecture & deep code · Claude Opus 5.5", 2, True),
    CuratedModelSpec("anthropic", "claude-sonnet-5", 200_000, "★ Coding workhorse · Claude Sonnet 5", 3, True),
    CuratedModelSpec("anthropic", "claude-opus-5", 200_000, "★ Frontier reasoning · Claude Opus 5", 4, True),
    CuratedModelSpec("anthropic", "claude-haiku-5-5", 200_000, "★ Fast coding · Claude Haiku 5.5", 5, False),
    CuratedModelSpec("anthropic", "claude-sonnet-4-6", 200_000, "★ Coding workhorse · Claude Sonnet 4.6", 6, True),
    CuratedModelSpec("anthropic", "claude-opus-4-8", 200_000, "★ Deep refactoring · Claude Opus 4.8", 7, False),
    CuratedModelSpec("anthropic", "claude-opus-4-7", 200_000, "Claude Opus 4.7", 8, False),
    CuratedModelSpec("anthropic", "claude-opus-4-6", 200_000, "Claude Opus 4.6", 9, False),
    CuratedModelSpec("anthropic", "claude-sonnet-4-5", 200_000, "Claude Sonnet 4.5", 10, False),
    CuratedModelSpec("anthropic", "claude-opus-4-5", 200_000, "Claude Opus 4.5", 11, False),
    CuratedModelSpec("anthropic", "claude-haiku-4-5", 200_000, "Claude Haiku 4.5 · fast & compact", 12, False),
    CuratedModelSpec("anthropic", "claude-fable-5-2", 200_000, "Claude Fable 5.2", 13, False),
    CuratedModelSpec("anthropic", "claude-fable-5-1", 200_000, "Claude Fable 5.1", 14, False),
    CuratedModelSpec("anthropic", "claude-fable-5", 200_000, "Claude Fable 5", 15, False),
    # OpenAI Direct
    CuratedModelSpec("openai", "gpt-5.3-codex", 400_000, "★ Agentic coding · GPT-5.3 Codex", 1, True),
    CuratedModelSpec("openai", "gpt-6.1-terra-pro", 1_050_000, "★ Frontier reasoning & code · GPT-6.1 Terra Pro", 2, True),
    CuratedModelSpec("openai", "gpt-6.1-terra", 1_050_000, "★ Frontier coding · GPT-6.1 Terra", 3, True),
    CuratedModelSpec("openai", "gpt-6.1-sol", 1_050_000, "★ Fast frontier coding · GPT-6.1 Sol", 4, False),
    CuratedModelSpec("openai", "gpt-6-astra", 1_050_000, "GPT-6 Astra · 1M context", 5, False),
    CuratedModelSpec("openai", "gpt-6-sol", 1_050_000, "GPT-6 Sol · 1M context", 6, False),
    CuratedModelSpec("openai", "gpt-6-luna", 1_050_000, "GPT-6 Luna · 1M context", 7, False),
    CuratedModelSpec("openai", "gpt-5.6-terra", 1_050_000, "GPT-5.6 Terra", 8, False),
    CuratedModelSpec("openai", "gpt-5.6-sol", 1_050_000, "GPT-5.6 Sol", 9, False),
    CuratedModelSpec("openai", "gpt-5.5", 1_050_000, "★ Frontier coding · GPT-5.5", 10, True),
    CuratedModelSpec("openai", "gpt-5.5-pro", 1_050_000, "GPT-5.5 Pro", 11, False),
    CuratedModelSpec("openai", "gpt-5.4", 1_050_000, "★ Coding & tool use · GPT-5.4", 12, True),
    CuratedModelSpec("openai", "gpt-5.4-pro", 1_050_000, "GPT-5.4 Pro", 13, False),
    CuratedModelSpec("openai", "gpt-5.4-mini", 400_000, "★ Fast coding · GPT-5.4 Mini", 14, False),
    CuratedModelSpec("openai", "gpt-5.4-nano", 400_000, "GPT-5.4 Nano", 15, False),
    CuratedModelSpec("openai", "gpt-5.2-codex", 400_000, "★ Agentic coding · GPT-5.2 Codex", 16, True),
    CuratedModelSpec("openai", "gpt-5.2", 400_000, "GPT-5.2", 17, False),
    CuratedModelSpec("openai", "gpt-5.1-codex-max", 400_000, "★ Long-horizon coding · GPT-5.1 Codex Max", 18, False),
    CuratedModelSpec("openai", "gpt-5.1-codex", 400_000, "GPT-5.1 Codex", 19, False),
    CuratedModelSpec("openai", "gpt-5.1-codex-mini", 400_000, "GPT-5.1 Codex Mini", 20, False),
    CuratedModelSpec("openai", "gpt-5-codex", 400_000, "GPT-5 Codex", 21, False),
    CuratedModelSpec("openai", "gpt-5", 400_000, "GPT-5", 22, False),
    CuratedModelSpec("openai", "gpt-5-mini", 400_000, "GPT-5 Mini", 23, False),
    CuratedModelSpec("openai", "codex-mini-latest", 200_000, "Codex Mini (latest)", 24, False),
    CuratedModelSpec("openai", "o4-mini", 200_000, "o4-mini reasoning", 25, False),
    CuratedModelSpec("openai", "o3", 200_000, "o3 reasoning", 26, False),
    CuratedModelSpec("openai", "o3-pro", 200_000, "o3-pro reasoning", 27, False),
    CuratedModelSpec("openai", "gpt-4.1", 1_047_576, "GPT-4.1 · 1M context", 28, False),
    CuratedModelSpec("openai", "gpt-4.1-mini", 1_047_576, "GPT-4.1 Mini · 1M context", 29, False),
    CuratedModelSpec("openai", "gpt-4o", 128_000, "GPT-4o", 30, False),
    CuratedModelSpec("openai", "gpt-4o-mini", 128_000, "GPT-4o Mini", 31, False),
    # Google Gemini Direct
    CuratedModelSpec("gemini", "gemini-4-hydrogen", 1_048_576, "★ 1M-ctx coding · Gemini 4 Hydrogen", 1, True),
    CuratedModelSpec("gemini", "gemini-4-carbon", 1_048_576, "★ 1M-ctx coding · Gemini 4 Carbon", 2, False),
    CuratedModelSpec("gemini", "gemini-4-argon", 1_048_576, "Gemini 4 Argon · 1M context", 3, False),
    CuratedModelSpec("gemini", "gemini-3.8-flash", 1_048_576, "★ Fast 1M-ctx coding · Gemini 3.8 Flash", 4, True),
    CuratedModelSpec("gemini", "gemini-3.7-flash", 1_048_576, "Gemini 3.7 Flash · 1M context", 5, False),
    CuratedModelSpec("gemini", "gemini-3.6-flash", 1_048_576, "Gemini 3.6 Flash · 1M context", 6, False),
    CuratedModelSpec("gemini", "gemini-3.5-flash", 1_048_576, "Gemini 3.5 Flash · 1M context", 7, False),
    CuratedModelSpec("gemini", "gemini-3.5-flash-lite", 1_048_576, "Gemini 3.5 Flash Lite", 8, False),
    CuratedModelSpec("gemini", "gemini-3.1-pro-preview", 1_048_576, "★ 1M-ctx architecture · Gemini 3.1 Pro", 9, True),
    CuratedModelSpec("gemini", "gemini-3.1-pro-preview-customtools", 1_048_576, "Gemini 3.1 Pro (custom tools)", 10, False),
    CuratedModelSpec("gemini", "gemini-3.1-flash-lite", 1_048_576, "Gemini 3.1 Flash Lite", 11, False),
    CuratedModelSpec("gemini", "gemini-3-pro-preview", 1_048_576, "Gemini 3 Pro Preview", 12, False),
    CuratedModelSpec("gemini", "gemini-3-flash-preview", 1_048_576, "Gemini 3 Flash Preview", 13, False),
    CuratedModelSpec("gemini", "gemini-2.5-pro", 1_048_576, "Gemini 2.5 Pro · 1M context", 14, False),
    CuratedModelSpec("gemini", "gemini-2.5-flash", 1_048_576, "Gemini 2.5 Flash · 1M context", 15, False),
    # DeepSeek Direct
    CuratedModelSpec("deepseek", "deepseek-v4.1-flash", 1_048_576, "★ Fast 1M-ctx coding · DeepSeek V4.1 Flash", 1, True),
    CuratedModelSpec("deepseek", "deepseek-v4-pro", 1_048_576, "★ Frontier reasoning & code · DeepSeek V4 Pro", 2, True),
    CuratedModelSpec("deepseek", "deepseek-v4-flash", 1_048_576, "★ High-speed coding · DeepSeek V4 Flash", 3, False),
    CuratedModelSpec("deepseek", "deepseek-v3.2", 163_840, "DeepSeek V3.2 · coding & agents", 4, False),
    CuratedModelSpec("deepseek", "deepseek-reasoner", 128_000, "DeepSeek Reasoner (R1)", 5, False),
    CuratedModelSpec("deepseek", "deepseek-chat", 128_000, "DeepSeek Chat (V3)", 6, False),
    # xAI Direct
    CuratedModelSpec("xai", "grok-build-0.1", 256_000, "★ Agentic engineering · Grok Build 0.1", 1, True),
    CuratedModelSpec("xai", "grok-4.7", 500_000, "★ Frontier coding · Grok 4.7", 2, True),
    CuratedModelSpec("xai", "grok-code-fast-1", 256_000, "★ Low-latency coding · Grok Code Fast 1", 3, True),
    CuratedModelSpec("xai", "grok-4.6", 500_000, "Grok 4.6 · 500k context", 4, False),
    CuratedModelSpec("xai", "grok-4.5", 500_000, "Grok 4.5 · 500k context", 5, False),
    CuratedModelSpec("xai", "grok-4.3", 1_000_000, "Grok 4.3 · 1M context", 6, False),
    CuratedModelSpec("xai", "grok-4.20-0309-reasoning", 2_000_000, "Grok 4.20 Reasoning · 2M context", 7, False),
    CuratedModelSpec("xai", "grok-4-1-fast-reasoning-latest", 256_000, "Grok 4.1 Fast Reasoning", 8, False),
    # Mistral Direct
    CuratedModelSpec("mistral", "devstral-2512", 262_144, "★ SWE agent model · Devstral 2512", 1, True),
    CuratedModelSpec("mistral", "devstral-latest", 262_144, "★ SWE agent model · Devstral Latest", 2, False),
    CuratedModelSpec("mistral", "codestral-2508", 256_000, "★ Code generation · Codestral 2508", 3, True),
    CuratedModelSpec("mistral", "codestral-latest", 256_000, "Codestral Latest", 4, False),
    CuratedModelSpec("mistral", "mistral-large-4-0", 1_048_576, "★ Frontier 1M-ctx · Mistral Large 4.0", 5, True),
    CuratedModelSpec("mistral", "mistral-large-2512", 262_144, "Mistral Large (2512)", 6, False),
    CuratedModelSpec("mistral", "mistral-medium-3-5", 262_144, "Mistral Medium 3.5", 7, False),
    CuratedModelSpec("mistral", "mistral-small-2603", 262_144, "Mistral Small (2603)", 8, False),
    # Together AI
    CuratedModelSpec("together", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", 262_144, "★ Open-weight coding · Qwen3 Coder 480B", 1, True),
    CuratedModelSpec("together", "Qwen/Qwen3.5-397B-A17B", 262_144, "Qwen3.5 397B A17B MoE", 2, False),
    CuratedModelSpec("together", "deepseek-ai/DeepSeek-V3.1", 163_840, "DeepSeek V3.1", 3, False),
    CuratedModelSpec("together", "deepseek-ai/DeepSeek-R1", 163_840, "DeepSeek R1", 4, False),
    # Groq
    CuratedModelSpec("groq", "moonshotai/kimi-k2-instruct-0905", 262_144, "★ Ultra-fast coding · Kimi K2 Instruct", 1, True),
    CuratedModelSpec("groq", "qwen/qwen3.6-27b", 262_144, "★ Ultra-fast coding · Qwen3.6 27B", 2, False),
    CuratedModelSpec("groq", "openai/gpt-oss-120b", 131_072, "OpenAI gpt-oss-120b on Groq", 3, False),
)

_PROVIDER_PREFIX: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "openrouter": "openrouter",
    "gemini": "gemini",
    "groq": "groq",
    "mistral": "mistral",
    "xai": "xai",
    "deepseek": "deepseek",
    "together": "together_ai",
}

ALL_CURATED_SPECS: tuple[CuratedModelSpec, ...] = (
    *(
        CuratedModelSpec("openrouter", model_id, ctx, tag, rank, quick)
        for model_id, ctx, tag, rank, quick in _OPENROUTER_SPECS
    ),
    *_DIRECT_PROVIDER_SPECS,
)

# OpenRouter router IDs whose first segment is literally "openrouter/..." on OpenRouter's API.
OPENROUTER_NATIVE_ROUTERS: frozenset[str] = frozenset(
    {
        "openrouter/auto",
        "openrouter/auto-code",
        "openrouter/auto-beta",
        "openrouter/pareto-code",
        "openrouter/free",
    }
)


def curated_catalog_overlay() -> dict[str, dict[str, Any]]:
    """Return LiteLLM-shaped catalog entries keyed by canonical route."""

    overlay: dict[str, dict[str, Any]] = {}
    for spec in ALL_CURATED_SPECS:
        prefix = _PROVIDER_PREFIX[spec.provider]
        route = f"{prefix}/{spec.model_id}"
        overlay[route] = _entry(prefix, spec.context_tokens, spec.tag, spec.rank)
    return overlay


def quick_pick_coding_models() -> tuple[CuratedModelSpec, ...]:
    """Featured coding models across providers for the one-step quick picker."""

    return tuple(spec for spec in ALL_CURATED_SPECS if spec.quick_pick)


_CONTEXT_BY_ROUTE: dict[str, int] = {}
for _spec in ALL_CURATED_SPECS:
    _prefix = _PROVIDER_PREFIX[_spec.provider]
    _CONTEXT_BY_ROUTE[f"{_prefix}/{_spec.model_id}"] = _spec.context_tokens
    if _spec.provider in {"openai", "anthropic"} or _spec.model_id in OPENROUTER_NATIVE_ROUTERS:
        _CONTEXT_BY_ROUTE.setdefault(_spec.model_id, _spec.context_tokens)


def known_context_window(model_name: str) -> int | None:
    """Return curated context window when LiteLLM's static table lacks a new model."""

    cleaned = model_name.strip()
    if cleaned in _CONTEXT_BY_ROUTE:
        return _CONTEXT_BY_ROUTE[cleaned]
    return None
