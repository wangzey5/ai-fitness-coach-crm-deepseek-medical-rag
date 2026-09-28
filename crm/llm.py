from __future__ import annotations

import json
import re
from typing import Any


INTENT_TOOL = {
    "type": "function",
    "name": "route_customer_intent",
    "description": "Route a CRM customer query to the correct internal workflow.",
    "parameters": {
        "type": "object",
        "properties": {
            "intent_code": {
                "type": "integer",
                "enum": [1, 2, 3, 4],
                "description": (
                    "1 recommend suitable exercises for a requested body part or condition, "
                    "2 infer training goal such as fat loss or muscle gain and build a plan, "
                    "3 assess height and weight with medical literature, 4 other"
                ),
            }
        },
        "required": ["intent_code"],
        "additionalProperties": False,
    },
    "strict": True,
}


class LLMService:
    def __init__(
        self,
        openai_api_key: str | None,
        openai_model: str,
        deepseek_api_key: str | None = None,
        deepseek_model: str = "deepseek-v4-flash",
        deepseek_base_url: str = "https://api.deepseek.com",
    ):
        self.provider = "local"
        self.model = openai_model
        self.client = None
        api_key = deepseek_api_key or openai_api_key
        base_url = deepseek_base_url if deepseek_api_key else None
        if deepseek_api_key:
            self.provider = "deepseek"
            self.model = deepseek_model
        elif openai_api_key:
            self.provider = "openai"
        if api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            except ImportError:
                self.client = None
                self.provider = "local"

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def classify_intent(self, query: str) -> int | None:
        if not self.client:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你负责路由健身私教 CRM 的中文请求。意图1是用户想知道自己该练什么动作、"
                            "某个部位该练什么、或有身体限制时适合什么运动；意图2是用户表达减脂、增肌、"
                            "塑形、提升体能等锻炼目的，需要配置训练计划；意图3是用户提供身高体重、BMI、"
                            "胖瘦或询问是否需要健身，需要健康评估和医学文献说明；"
                            "其余为意图4。只输出 JSON，例如 {\"intent_code\":1}，不要输出解释。"
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            parsed = self._parse_json(content)
            intent_code = int(parsed.get("intent_code", 0))
            if intent_code in {1, 2, 3, 4}:
                return intent_code
        except Exception:
            return None
        return None

    def answer(self, query: str, context: str, history: list[dict[str, str]], intent_code: int = 4) -> str | None:
        if not self.client:
            return None
        intent_names = {
            1: "动作搜索推荐",
            2: "锻炼目的与计划配置",
            3: "身高体重健康评估",
            4: "其他健身咨询",
        }
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是健身私教 CRM 的问询式顾问。回答必须循序渐进，不要一次性堆出完整长方案。"
                    "先针对客户已经说出的目标给出简短、具体的判断；如果训练经验、器械条件、伤病限制、"
                    "单次时长或恢复情况不足，只追问 1 到 2 个最关键问题。资料足够时，再给一个可执行的"
                    "下一步方案，最多 3 条。只选取资料中最相关的 1 到 2 点回答，不要把知识库全部复述。"
                    "涉及疼痛、疾病、伤病恢复或高风险动作时，提醒客户先咨询医生或现场专业教练。"
                ),
            }
        ]
        messages.extend(history[-8:])
        messages.append(
            {
                "role": "user",
                "content": (
                    f"当前意图：{intent_names.get(intent_code, '其他健身咨询')}\n"
                    f"可参考资料：\n{context}\n\n"
                    f"客户问题：{query}\n\n"
                    "请直接回复客户，语气像私教沟通。"
                ),
            }
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
                max_tokens=450,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return None

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return {}
        return {}
