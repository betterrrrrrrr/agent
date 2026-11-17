param([string[]]$TestPaths = @('tests/test_ch07_layers.py', 'tests/test_ch07_storage.py', 'tests/test_summarizer.py', 'tests/test_memory.py', 'tests/test_context_assembly.py', 'tests/test_prompt_cache_prefix.py', 'tests/test_config.py', 'tests/test_llm.py', 'tests/test_schemas.py', 'tests/graph', 'tests/tools', 'tests/test_repository.py', 'tests/test_conversations_api.py', 'tests/test_chat_api.py', 'tests/test_agent_api.py', 'tests/test_prompts.py'))
$ErrorActionPreference = 'Stop'
# Offline test credentials only; restore the caller's environment after testing.
$ch07TestEnv = @{
    CHAT_MODEL='deepseek-ai/DeepSeek-V4-Flash'
    CHAT_BASE_URL='http://127.0.0.1:9/v1'
    CHAT_API_KEY='offline-test'
    EMBED_API_KEY='offline-test'
    RERANK_API_KEY='offline-test'
    PYTHONIOENCODING='utf-8'
}
$ch07SavedEnv = @{}
foreach ($key in $ch07TestEnv.Keys) {
    $ch07SavedEnv[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    [Environment]::SetEnvironmentVariable($key, $ch07TestEnv[$key], 'Process')
}
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    & "$PSScriptRoot\..\.venv\Scripts\python.exe" -m pytest @TestPaths --offline-db -q --disable-warnings
    $ch07TestExit = $LASTEXITCODE
} finally {
    Pop-Location
    foreach ($key in $ch07SavedEnv.Keys) {
        [Environment]::SetEnvironmentVariable($key, $ch07SavedEnv[$key], 'Process')
    }
}
exit $ch07TestExit
