# 三层上下文与动态预算

## 数据库升级

本次仅修改 daima。生产继续使用 MySQL，未自动连接或修改任何现存数据库。
已应用 `sql/ch07-ddl.sql` 的库，在备份后运行 `sql/ch07-layers.sql` 一次；全新库先按 README 建表，再依次应用两个文件。
在 MySQL 客户端选定 mewhelp 库后执行：

```sql
SOURCE D:/xiaolin_mewhelp/daima/sql/ch07-layers.sql;
```

脚本新增原文边界和层2实际用量、建立分段表，并把原滚动摘要迁为首段。DDL 会自动提交，脚本的 ALTER 不可重复运行。
旧消息、checkpoint、会话侧栏保持完整；新请求读取两个边界。摘要和检索证据只在本轮临时消息中注入。

## 预算

`python -m scripts.context_budget` 打印当前配置；需要先在 `.env` 填齐模型账户配置，但该命令不调用上游。
统一用 `ceil(字符数 / ZH_CHARS_PER_TOKEN)` 和 LangChain 近似计数，默认中文口径为1。
这是近似预算，服务端实际 tokenizer 与通道窗口仍需人工校准。

```text
fixed = system/schema估计 + TopK证据预算 + 最近K段摘要预算 + 输出上限 + 安全余量
peak  = 单轮用户上限 + (最大步数-1) × (单步输出上限 + 最大工具调用数 × 单工具结果上限)
history = min(目标轮数 × (用户上限+最终输出上限), 窗口-fixed-peak)
layer1 = floor(history × 0.7)
layer2 = history-layer1
```

`CONTEXT_WINDOW_MAX_TOKENS>0` 只收紧历史预算；不会跳过窗口约束。
每次 Agent 调用前还会把实际工具 schema 和全部消息计入复核，只在完整用户轮边界继续裁旧轮；当前轮也装不下时拒绝上游调用。
工具结果截短，模型输出通过 API max_tokens 约束；超长用户输入在落库前拒绝。

## 演示

在 daima 中运行：

```powershell
.\scripts\demo_ch07.ps1
```

18,000 窗口、2,000输入/输出、3步、每步1次工具、单工具1,200、TopK5：
fixed=6350，peak=8400，history=3250，layer1=2275，layer2=975。
比文档的5650更保守，因为按每一步都可能输出到API上限预留，不用未约束的平均AI长度估算。
`max_tool_calls_per_step` 限制实际执行数，超出部分仍以错误 ToolMessage 逐一回应，不删掉模型的调用记录。
启动后打开 http://127.0.0.1:8000 ，连续对话中观察：

```powershell
Get-Content .\log\app.log -Wait | Select-String 'context_budget|层1 降级|summary |model_ctx|history_ctx'
```

先出现层1降级，层2用量超过预算才触发后台任务；任务完成追加一段并推进摘要边界。下一轮才加载新摘要。
默认大窗口下二十轮短对话应无降级和摘要。
摘要只保留最近3段投影用于模型输入，更早段仍保存在分段表；不提供跨会话长期记忆。

## 验证

```powershell
.\scripts\check_ch07.ps1
```

离线使用内存 SQLite 和模型替身验证仓储事务、36轮级联、并发调度和相邻回归，不连接任何现存数据库或模型账号。
当前定向与相邻回归共228项通过，包含流式/非流式各24轮的真实LangGraph状态接线验证。
MySQL 行锁的跨进程行为需在 MySQL 环境复验，SQLite 测试不代表该行为已获验证。
真实摘要质量用配置好上游后的 `python -m scripts.eval_ch07` 验证（会产生模型调用费用）。
