$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
# 需先配置 .env 的真实模型账号并应用 sql/ch07-layers.sql。
$env:MODEL_CONTEXT_WINDOW = '18000'
$env:MAX_OUTPUT_TOKENS = '2000'
$env:MAX_USER_INPUT_TOKENS = '2000'
$env:MAX_AGENT_STEPS = '3'
$env:TOOL_RESULT_MAX_TOKENS = '1200'
$env:RERANK_TOP_K = '5'
$env:CONTEXT_WINDOW_MAX_TOKENS = '0'
& "$PSScriptRoot\..\.venv\Scripts\python.exe" -m scripts.context_budget
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& "$PSScriptRoot\..\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
