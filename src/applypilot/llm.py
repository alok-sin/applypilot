"""Unified LLM client for ApplyPilot using LiteLLM.

Runtime contract:
  - If set, LLM_MODEL must be a fully-qualified LiteLLM model string
    (for example: openai/gpt-4o-mini, anthropic/claude-3-5-haiku-latest,
    gemini/gemini-3.0-flash).
  - If LLM_MODEL is unset, provider is inferred by first configured source:
    GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, then LLM_URL.
  - Credentials come from provider env vars or generic LLM_API_KEY.
  - LLM_URL is optional for custom OpenAI-compatible endpoints.
  - LLM_STREAMING_MODE enables streaming mode for LLM proxies that require it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack
import warnings

import litellm

if TYPE_CHECKING:
    from applypilot.core.context import RunContext

# Suppress pydantic serialization warnings from litellm internals when provider
# responses have fewer fields than the full ModelResponse schema.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")

log = logging.getLogger(__name__)

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds
_INFERRED_SOURCE_ORDER: tuple[tuple[str, str], ...] = (
    ("gemini", "GEMINI_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("lightning", "LIGHTNING_API_KEY"),
    ("openai", "LLM_URL"),
)
_DEFAULT_MODEL_BY_PROVIDER = {
    "gemini": "gemini/gemini-3.0-flash",
    "openai": "openai/gpt-5-mini",
    "anthropic": "anthropic/claude-haiku-4-5",
    "lightning": "openai/lightning-ai/gemma-4-31B-it",
    "gemini-cli": "gemini-cli/gemini-3.1-pro-preview",
}
_DEFAULT_LOCAL_MODEL = "openai/local-model"
_DEFAULT_FALLBACK_MODEL = "gemini/gemini-2.5-flash-preview-04-17"

# Auto-detected api_base URLs for providers that need them.
_PROVIDER_API_BASE = {
    "lightning": "https://lightning.ai/api/v1",
}


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration consumed by LLMClient."""

    provider: str
    api_base: str | None
    model: str
    api_key: str
    use_streaming: bool = False
    fallback_model: str | None = None
    fallback_api_key: str | None = None
    fallback_api_base: str | None = None


class ChatMessage(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # Optional cache hint. On Anthropic this becomes a cache_control breakpoint
    # on the resulting content block. On other providers the key is stripped
    # and identical prefixes are cached implicitly (Gemini/OpenAI) or not at
    # all (local/Lightning/gemini-cli).
    cache: Literal["ephemeral"]


def _apply_cache_markers(provider: str, messages: list) -> list:
    """Transform the optional `cache` hint on messages into provider-specific form.

    For Anthropic, a message like `{role, content: str, cache: "ephemeral"}`
    becomes `{role, content: [{type: "text", text: str, cache_control: {type: "ephemeral"}}]}`.
    For every other provider the `cache` key is stripped and `content` stays
    a plain string.
    """
    out: list = []
    for m in messages:
        if "cache" not in m:
            out.append(m)
            continue
        stripped = {k: v for k, v in m.items() if k != "cache"}
        if provider == "anthropic":
            content = stripped.get("content", "")
            if isinstance(content, str):
                stripped["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
        out.append(stripped)
    return out


class LiteLLMExtra(TypedDict, total=False):
    stop: str | list[str]
    top_p: float
    seed: int
    stream: bool
    response_format: dict[str, Any]
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, Any]
    fallbacks: list[str]


def _env_get(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _provider_from_model(model: str) -> str:
    provider, _, model_name = model.partition("/")
    if not provider or not model_name:
        raise RuntimeError("LLM_MODEL must include a provider prefix (for example 'openai/gpt-4o-mini').")
    return provider


def _infer_provider_and_source(env: Mapping[str, str]) -> tuple[str, str] | None:
    for provider, env_key in _INFERRED_SOURCE_ORDER:
        if _env_get(env, env_key):
            return provider, env_key
    return None


def resolve_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """Resolve LLM configuration from environment."""
    env_map = env if env is not None else os.environ

    model = _env_get(env_map, "LLM_MODEL")
    local_url = _env_get(env_map, "LLM_URL")
    inferred = _infer_provider_and_source(env_map)
    if model:
        if "/" in model:
            provider = _provider_from_model(model)
        elif inferred:
            provider, _ = inferred
            model = f"{provider}/{model}"
        else:
            raise RuntimeError("LLM_MODEL must include a provider prefix (for example 'openai/gpt-4o-mini').")
    else:
        if not inferred:
            raise RuntimeError(
                "No LLM provider configured. Set one of GEMINI_API_KEY, OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, LLM_URL, or LLM_MODEL."
            )
        provider, source = inferred
        if source == "LLM_URL":
            model = _DEFAULT_LOCAL_MODEL
        else:
            model = _DEFAULT_MODEL_BY_PROVIDER[provider]

    provider_api_key_env = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "lightning": "LIGHTNING_API_KEY",
    }
    api_key_env = provider_api_key_env.get(provider, "LLM_API_KEY")
    api_key = _env_get(env_map, api_key_env) or _env_get(env_map, "LLM_API_KEY")

    # gemini-cli uses the CLI's own auth — no API key needed.
    if not api_key and not local_url and provider != "gemini-cli":
        key_help = f"{api_key_env} or LLM_API_KEY" if provider in provider_api_key_env else "LLM_API_KEY"
        raise RuntimeError(
            f"Missing credentials for LLM_MODEL '{model}'. Set {key_help}, or set LLM_URL for "
            "a local OpenAI-compatible endpoint."
        )

    # Check if streaming mode is enabled via environment variable
    use_streaming = _env_get(env_map, "LLM_STREAMING_MODE").lower() in ("true", "1", "yes")

    # Resolve api_base: explicit LLM_URL takes precedence, then provider defaults.
    api_base = local_url.rstrip("/") if local_url else _PROVIDER_API_BASE.get(provider)

    # Resolve fallback model from env or use default.
    fallback_model = _env_get(env_map, "LLM_FALLBACK_MODEL") or _DEFAULT_FALLBACK_MODEL
    fallback_provider = _provider_from_model(fallback_model)
    fallback_api_key_env = provider_api_key_env.get(fallback_provider, "LLM_API_KEY")
    fallback_api_key = _env_get(env_map, fallback_api_key_env) or _env_get(env_map, "LLM_API_KEY") or api_key
    fallback_api_base = _PROVIDER_API_BASE.get(fallback_provider)

    return LLMConfig(
        provider=provider,
        api_base=api_base,
        model=model,
        api_key=api_key,
        use_streaming=use_streaming,
        fallback_model=fallback_model,
        fallback_api_key=fallback_api_key,
        fallback_api_base=fallback_api_base,
    )


def _call_gemini_cli(model: str, messages: list[ChatMessage], timeout: int) -> str:
    """Invoke the `gemini` CLI as a completion backend.

    Messages are flattened into a single prompt. The CLI prints the model
    response to stdout. Raises RuntimeError on non-zero exit or empty output.
    """
    import shutil
    import subprocess

    if shutil.which("gemini") is None:
        raise RuntimeError(
            "gemini CLI not found on PATH. Install from https://github.com/google-gemini/gemini-cli"
        )

    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        if role == "SYSTEM":
            parts.append(f"[SYSTEM INSTRUCTIONS]\n{content}")
        else:
            parts.append(f"[{role}]\n{content}")
    prompt = "\n\n".join(parts)

    # model is "gemini-cli/gemini-3.1-pro-preview" — strip prefix for -m flag.
    _, _, model_name = model.partition("/")

    # Always pass -m explicitly so the CLI doesn't fall back to its config default.
    cmd = ["gemini", "-m", model_name, "-p", prompt]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"gemini CLI timed out after {timeout}s") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:500]
        raise RuntimeError(f"gemini CLI exit {result.returncode}: {stderr}")

    text = (result.stdout or "").strip()
    if not text:
        raise RuntimeError("gemini CLI returned empty output")
    return text


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if *exc* originated from an HTTP 429 / rate-limit response."""
    # litellm raises litellm.RateLimitError for 429s.
    if type(exc).__name__ == "RateLimitError":
        return True
    # Some wrappers nest the real error; check status_code attribute.
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    # Check the inner cause as well.
    if exc.__cause__ and exc.__cause__ is not exc:
        return _is_rate_limit_error(exc.__cause__)
    return False


class LLMClient:
    """Thin wrapper around LiteLLM completion()."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.provider = config.provider
        self.model = config.model
        self._use_streaming = config.use_streaming
        litellm.suppress_debug_info = True

    def _do_completion(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        api_key: str | None,
        api_base: str | None,
        max_output_tokens: int,
        temperature: float | None,
        timeout: int,
        num_retries: int,
        drop_params: bool,
        **extra: Unpack[LiteLLMExtra],
    ) -> str:
        """Run a single litellm.completion() call and return text."""
        # CLI backend: shell out instead of calling litellm.
        if model.startswith("gemini-cli/"):
            return _call_gemini_cli(model, messages, timeout)

        provider = _provider_from_model(model)
        prepared = _apply_cache_markers(provider, list(messages))

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": prepared,
            "max_tokens": max_output_tokens,
            "timeout": timeout,
            "num_retries": num_retries,
            "drop_params": drop_params,
            "api_key": api_key,
            "api_base": api_base,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        kwargs.update(extra)

        response = litellm.completion(**kwargs)

        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("LLM response contained no choices.")
        content = response.choices[0].message.content
        text = content.strip() if isinstance(content, str) else str(content).strip()
        if not text:
            raise RuntimeError("LLM response contained no text content.")
        return text

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int = 10000,
        temperature: float | None = None,
        timeout: int = _TIMEOUT,
        num_retries: int = _MAX_RETRIES,
        drop_params: bool = True,
        **extra: Unpack[LiteLLMExtra],
    ) -> str:
        """Send a completion request and return plain text content."""
        # Use streaming mode when configured (required by some LLM proxies)
        if self._use_streaming:
            return self._chat_streaming(
                messages=messages,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                num_retries=num_retries,
                drop_params=drop_params,
                **extra,
            )

        # Standard non-streaming call with optional fallback.
        from applypilot.cancellation import stop_event

        if stop_event.is_set():
            raise KeyboardInterrupt()
        try:
            return self._do_completion(
                messages,
                model=self.model,
                api_key=self.config.api_key or None,
                api_base=self.config.api_base or None,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout=timeout,
                num_retries=num_retries,
                drop_params=drop_params,
                **extra,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as primary_exc:
            # litellm can wrap a KeyboardInterrupt from httpx into a generic
            # exception. If cancellation was requested, don't fall back.
            if stop_event.is_set():
                raise KeyboardInterrupt() from primary_exc
            if not self.config.fallback_model:
                raise RuntimeError(f"LLM request failed ({self.provider}/{self.model}): {primary_exc}") from primary_exc

            is_rate_limit = _is_rate_limit_error(primary_exc)
            if is_rate_limit:
                log.warning(
                    "Primary model %s hit rate limit (429), switching to fallback %s",
                    self.model, self.config.fallback_model,
                )
            else:
                log.warning(
                    "Primary model %s failed (%s), falling back to %s",
                    self.model, primary_exc, self.config.fallback_model,
                )
            try:
                return self._do_completion(
                    messages,
                    model=self.config.fallback_model,
                    api_key=self.config.fallback_api_key or self.config.api_key or None,
                    api_base=self.config.fallback_api_base or None,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    num_retries=num_retries,
                    drop_params=drop_params,
                    **extra,
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"LLM request failed: primary ({self.model}): {primary_exc}; "
                    f"fallback ({self.config.fallback_model}): {fallback_exc}"
                ) from fallback_exc

    def _chat_streaming(
        self,
        messages: list[ChatMessage],
        *,
        max_output_tokens: int = 10000,
        temperature: float | None = None,
        num_retries: int = _MAX_RETRIES,
        drop_params: bool = True,
        **extra: Unpack[LiteLLMExtra],
    ) -> str:
        """Use streaming completion mode.

        Some LLM proxies require streaming mode. This method uses stream=True
        and accumulates the chunks into a plain text response.
        """
        try:
            prepared = _apply_cache_markers(self.provider, list(messages))
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": prepared,
                "max_tokens": max_output_tokens,
                "num_retries": num_retries,
                "drop_params": drop_params,
                "api_key": self.config.api_key or None,
                "api_base": self.config.api_base or None,
                "stream": True,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature

            response = litellm.completion(**kwargs)

            # Accumulate content from streaming chunks
            content_parts = []
            for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content_parts.append(delta.content)

            text = "".join(content_parts).strip()

            if not text:
                raise RuntimeError("LLM response contained no text content.")
            return text
        except Exception as exc:
            raise RuntimeError(f"LLM request failed ({self.provider}/{self.model}): {exc}") from exc

    def close(self) -> None:
        """No-op. LiteLLM completion() is stateless per call."""
        return None


_clients: dict[str, LLMClient] = {}
_lock = threading.Lock()
_llm_yaml_cache: dict | None = None
_env_loaded = False


def _load_llm_yaml() -> dict | None:
    """Load and cache ~/.applypilot/llm.yaml if it exists."""
    global _llm_yaml_cache
    if _llm_yaml_cache is not None:
        return _llm_yaml_cache if _llm_yaml_cache else None
    try:
        from applypilot.config import LLM_CONFIG_PATH

        if LLM_CONFIG_PATH.exists():
            import yaml

            with open(LLM_CONFIG_PATH, encoding="utf-8") as f:
                _llm_yaml_cache = yaml.safe_load(f) or {}
            log.info("Loaded per-task LLM config from %s", LLM_CONFIG_PATH)
            return _llm_yaml_cache
    except Exception as exc:
        log.warning("Failed to load llm.yaml: %s", exc)
    _llm_yaml_cache = {}
    return None


_PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "lightning": "LIGHTNING_API_KEY",
}


def _task_config_from_yaml(
    yaml_cfg: dict | None,
    env: Mapping[str, str],
    task: str,
) -> LLMConfig | None:
    """Pure: build an LLMConfig from a parsed llm.yaml dict + env mapping.

    Returns ``None`` if the yaml doesn't configure this task (caller should
    fall through to ``resolve_llm_config(env)`` or its own default).
    """
    if not yaml_cfg:
        return None

    # Look up task-specific config, fall back to "default" section.
    tasks = yaml_cfg.get("tasks") or {}
    task_entry = tasks.get(task)
    if not task_entry and task != "default":
        task_entry = yaml_cfg.get("default")
    if not task_entry:
        return None

    model = task_entry.get("model")
    if not model:
        return None

    provider = _provider_from_model(model)

    # Resolve API key: task-level api_key_env → provider default → LLM_API_KEY
    api_key_env_name = task_entry.get("api_key_env") or _PROVIDER_KEY_ENV.get(provider, "LLM_API_KEY")
    api_key = _env_get(env, api_key_env_name) or _env_get(env, "LLM_API_KEY")

    # Resolve api_base: task-level → provider default → LLM_URL
    api_base = task_entry.get("api_base") or _PROVIDER_API_BASE.get(provider) or _env_get(env, "LLM_URL")
    api_base = api_base.rstrip("/") if api_base else None

    use_streaming = str(task_entry.get("streaming", "")).lower() in ("true", "1", "yes")

    # Resolve fallback model: task-level → global default.
    fallback_model = task_entry.get("fallback") or _DEFAULT_FALLBACK_MODEL
    fallback_provider = _provider_from_model(fallback_model)
    # Explicit fallback_api_key_env takes priority (needed for cross-provider fallbacks).
    fb_api_key_env = task_entry.get("fallback_api_key_env") or _PROVIDER_KEY_ENV.get(fallback_provider, "LLM_API_KEY")
    fallback_api_key = _env_get(env, fb_api_key_env) or _env_get(env, "LLM_API_KEY") or api_key
    fallback_api_base = task_entry.get("fallback_api_base") or _PROVIDER_API_BASE.get(fallback_provider)

    return LLMConfig(
        provider=provider,
        api_base=api_base,
        model=model,
        api_key=api_key,
        use_streaming=use_streaming,
        fallback_model=fallback_model,
        fallback_api_key=fallback_api_key,
        fallback_api_base=fallback_api_base,
    )


def _resolve_task_config(task: str) -> LLMConfig | None:
    """Module-global wrapper — reads the cached llm.yaml and ``os.environ``.

    Kept for today's ``get_client()`` callers; new code should prefer
    :func:`resolve_config_for_ctx`, which scopes resolution to a
    :class:`RunContext` instead of process state.
    """
    return _task_config_from_yaml(_load_llm_yaml(), os.environ, task)


def _ctx_env(ctx: "RunContext") -> dict[str, str]:
    """Snapshot env for this ctx: ``os.environ`` shadowed by user secrets.

    ``SecretsProvider.get`` is authoritative where it returns a value
    (matches today's ``.env``-beats-process-env precedence). Fall through
    to ``os.environ`` for keys the provider doesn't know.
    """
    env = dict(os.environ)
    secrets = ctx.user.secrets
    if secrets is None:
        return env
    # Pre-seed provider/LLM keys so the pure resolvers see them as regular
    # mapping entries. Only overwrites when the provider has a concrete
    # value — missing secrets leave os.environ's reading intact.
    for key in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LIGHTNING_API_KEY",
        "LLM_API_KEY",
        "LLM_URL",
        "LLM_MODEL",
        "LLM_FALLBACK_MODEL",
        "LLM_STREAMING_MODE",
    ):
        value = secrets.get(key)
        if value:
            env[key] = value
    return env


def resolve_config_for_ctx(ctx: "RunContext", task: str = "default") -> LLMConfig:
    """Resolve the effective LLM config for this ctx + task.

    Order:
      1. ``ctx.task.llm_overrides[task]`` — per-task BYO config.
      2. ``ctx.user.llm_config`` (parsed ``llm.yaml``) + user's secrets/env.
      3. Env-only ``resolve_llm_config`` (today's default path).
    """
    override = ctx.task.llm_overrides.get(task)
    if override is not None:
        return override

    env = _ctx_env(ctx)
    from_yaml = _task_config_from_yaml(ctx.user.llm_config, env, task)
    if from_yaml is not None:
        return from_yaml

    return resolve_llm_config(env)


def get_client_for_ctx(ctx: "RunContext", task: str = "default") -> LLMClient:
    """Build a fresh :class:`LLMClient` for this ctx + task.

    Unlike :func:`get_client`, this does not use the process-global
    ``_clients`` cache — two ctxs (two users) must never share a client
    that was built with someone else's credentials. Workers that reuse a
    ctx across tasks can cache on their own side.
    """
    config = resolve_config_for_ctx(ctx, task)
    return LLMClient(config)


def get_client(task: str = "default") -> LLMClient:
    """Return (or create) an LLMClient for the given task.

    When llm.yaml exists and defines the task, uses that config.
    Otherwise falls back to the global env-var based config.
    Clients are cached per task key.
    """
    global _env_loaded
    if task not in _clients:
        with _lock:
            if task not in _clients:
                if not _env_loaded:
                    try:
                        from applypilot.config import load_env

                        load_env()
                    except ModuleNotFoundError:
                        log.debug("python-dotenv not installed; skipping .env auto-load in llm.get_client().")
                    _env_loaded = True

                # Try task-specific config from llm.yaml first.
                config = _resolve_task_config(task)
                if config is None and task != "default":
                    # No task-specific config — share the default client.
                    default_client = get_client("default")
                    _clients[task] = default_client
                    return default_client
                if config is None:
                    # No llm.yaml at all — use env-var resolution (original behavior).
                    config = resolve_llm_config()

                log.info("LLM [%s] provider: %s  model: %s", task, config.provider, config.model)
                _clients[task] = LLMClient(config)
    return _clients[task]
