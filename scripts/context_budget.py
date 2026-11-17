"""打印当前配置的窗口预算：python -m scripts.context_budget。无需调用模型或数据库。"""
from dataclasses import asdict
import json
from app.core.budget import compute


if __name__ == '__main__':
    print(json.dumps(asdict(compute()), ensure_ascii=False, indent=2))
