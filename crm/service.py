from __future__ import annotations

import re
from typing import Any

from .database import Database
from .llm import LLMService
from .sql_agent import ReadOnlySQLAgent
from .vector_store import VectorStore
from .web_search import WebSearchService


class ChatService:
    def __init__(
        self,
        database: Database,
        vector_store: VectorStore,
        sql_agent: ReadOnlySQLAgent,
        llm: LLMService,
        search_service: WebSearchService,
    ):
        self.database = database
        self.vector_store = vector_store
        self.sql_agent = sql_agent
        self.llm = llm
        self.search_service = search_service

    def detect_intent_code(self, query: str) -> int:
        lowered = query.lower()
        if self._extract_body_metrics(query):
            return 3
        if any(word in lowered for word in ("身高", "体重", "bmi", "胖", "肥胖", "超重", "要不要健身", "需不需要健身")):
            return 3
        if any(word in lowered for word in ("减脂", "减肥", "增肌", "塑形", "变瘦", "变壮", "马甲线", "体脂", "训练计划", "健身计划")):
            return 2
        if self._specific_exercise(query) or any(word in lowered for word in ("注意", "要点", "怎么做", "标准", "发力", "错误", "禁忌")):
            return 1
        if any(
            word in lowered
            for word in (
                "练",
                "锻炼",
                "动作",
                "运动",
                "划船",
                "下拉",
                "深蹲",
                "臀桥",
                "背",
                "胸",
                "腿",
                "肩",
                "臀",
                "腹",
                "核心",
                "腰",
                "膝",
                "颈椎",
                "不舒服",
                "疼",
                "痛",
            )
        ):
            return 1
        llm_intent = self.llm.classify_intent(query)
        return llm_intent or 4

    def interview(
        self,
        customer_id: str,
        query: str,
        name: str | None = None,
        phone_tail: str | None = None,
    ) -> dict[str, Any]:
        user, is_new = self.database.ensure_user(customer_id, name=name, phone_tail=phone_tail)
        history = self.database.fetch_chat_history(customer_id)
        intent_code = self.detect_intent_code(query)

        if intent_code == 1:
            response = self._exercise_recommendation_answer(query, history)
        elif intent_code == 2:
            response = self._training_goal_plan_answer(query, history)
        elif intent_code == 3:
            response = self._health_assessment_answer(query)
        else:
            response = (
                "我可以帮你做三件事：\n"
                "1. 你说想练背、练胸、腰不好等，我联网搜索并推荐合适动作。\n"
                "2. 你说减脂、增肌、塑形等目标，我用 DeepSeek 配训练计划。\n"
                "3. 你输入身高体重，我用医学文献 RAG 判断是否建议运动。"
            )

        if is_new:
            response = "你好，欢迎使用 A3 健身私教 CRM。" + response
        self.database.update_user_intent(customer_id, intent_code)
        self.database.append_exchange(customer_id, query, response)
        updated_user = self.database.get_user(customer_id) or user
        return {
            "phone_number": customer_id,
            "intent_code": intent_code,
            "is_new_user": is_new,
            "response": response,
            "user": {**updated_user, "last_intent": intent_code},
        }

    def _exercise_recommendation_answer(self, query: str, history: list[dict[str, str]]) -> str:
        exercise = self._specific_exercise(query) or self._referenced_previous_exercise(query, history)
        if exercise:
            return self._exercise_detail_answer(exercise, query, history)

        context = self._exercise_context(query, history)
        profile = self._exercise_profile(query, context)
        target = profile["target"]
        search_results = self.search_service.search_exercises(f"{target} 适合动作", limit=3)
        exercises = self._exercise_options(profile)
        sources = "\n".join(
            f"- {item.title}（{item.source}）：{item.url}" for item in search_results
        )
        llm_context = (
            f"用户训练画像：{profile}\n"
            f"搜索来源：\n{sources}\n"
            "请根据画像推荐 3 个动作。不同场景、器械、疼痛限制必须给不同建议；不要机械复述固定模板。"
        )
        generated = self.llm.answer(query, llm_context, history, intent_code=1)
        if generated:
            return generated
        setting = profile["setting"]
        constraints = self._profile_sentence(profile)
        setting_note = f"我也记住了你的条件：{setting}。" if setting else "下一步只确认一个点：你是在家练还是健身房练？"
        return (
            f"我先判断你现在要解决的是：{target}适合练什么。{constraints}\n\n"
            "直接推荐这几个动作，先从低风险版本开始：\n"
            + "\n".join(f"{index}. {exercise}" for index, exercise in enumerate(exercises, start=1))
            + "\n\n"
            f"{setting_note}我可以继续按器械条件把动作再缩成 3 个最合适的。"
        )

    def _exercise_detail_answer(self, exercise: str, query: str, history: list[dict[str, str]]) -> str:
        question_type = self._exercise_question_type(query)
        search_query = f"{exercise} 怎么做 动作解释" if question_type == "explain" else f"{exercise} 动作要点 注意事项"
        search_results = self.search_service.search_exercises(search_query, limit=3)
        sources = "\n".join(
            f"- {item.title}（{item.source}）：{item.url}" for item in search_results
        )
        details = self._exercise_explanation(exercise) if question_type == "explain" else self._exercise_detail_tips(exercise)
        instruction = (
            "请解释这个动作是什么、练哪里、怎么做，不要只讲注意事项。"
            if question_type == "explain"
            else "请直接回答该动作注意事项，不要重新推荐一组动作。"
        )
        llm_context = (
            f"用户追问的具体动作：{exercise}\n"
            f"问题类型：{question_type}\n"
            f"本地参考：\n" + "\n".join(details) + "\n"
            f"搜索来源：\n{sources}\n"
            f"{instruction}"
        )
        generated = self.llm.answer(query, llm_context, history, intent_code=1)
        if generated:
            return generated
        if question_type == "explain":
            return (
                f"你问的是「{exercise}」是什么，我先解释动作本身：\n"
                + "\n".join(f"{index}. {detail}" for index, detail in enumerate(details, start=1))
                + "\n\n等你知道动作怎么做之后，再需要我可以继续讲它的注意点和常见错误。"
            )
        return (
            f"你问的是「{exercise}」的注意点，不是新的训练目标。我直接说动作要点：\n"
            + "\n".join(f"{index}. {tip}" for index, tip in enumerate(details, start=1))
            + f"\n\n{self._exercise_safety_fallback(exercise)}"
        )

    def _exercise_context(self, query: str, history: list[dict[str, str]]) -> str:
        user_messages = [item["content"] for item in history if item["role"] == "user"]
        return " ".join(user_messages[-3:] + [query])

    def _specific_exercise(self, query: str) -> str | None:
        lowered = query.lower()
        exercises = (
            "毛巾划船",
            "弹力带划船",
            "坐姿划船",
            "俯身划船",
            "高位下拉",
            "箱式深蹲",
            "哑铃杯式深蹲",
            "臀桥",
            "台阶上步",
            "死虫",
            "鸟狗",
            "侧桥",
            "俯身 y-t-w",
            "超人式背伸",
        )
        for exercise in exercises:
            if exercise.lower() in lowered:
                return exercise
        if "划船" in lowered:
            return "划船"
        if "深蹲" in lowered:
            return "深蹲"
        return None

    def _referenced_previous_exercise(self, query: str, history: list[dict[str, str]]) -> str | None:
        lowered = query.lower()
        if not any(word in lowered for word in ("第一个", "第二个", "第三个", "这个动作", "它", "注意")):
            return None
        recent_assistant = " ".join(item["content"] for item in history[-4:] if item["role"] == "assistant")
        matches = re.findall(r"\d+\.\s*([^：:\n]+)", recent_assistant)
        if "第二个" in lowered and len(matches) >= 2:
            return matches[1].strip()
        if "第三个" in lowered and len(matches) >= 3:
            return matches[2].strip()
        if matches:
            return matches[0].strip()
        return None

    def _exercise_question_type(self, query: str) -> str:
        lowered = query.lower()
        if any(word in lowered for word in ("是什么", "什么是", "怎么做", "如何做", "不会做", "教我", "解释")):
            return "explain"
        return "caution"

    def _exercise_explanation(self, exercise: str) -> list[str]:
        if "死虫" in exercise:
            return [
                "死虫是一个核心稳定训练，不是练腹肌卷起来，而是练你在手脚移动时让腰背保持稳定。",
                "基本做法：仰卧，双手指向天花板，髋和膝大约弯曲 90 度，先让腰背轻轻贴近地面。",
                "开始动作：慢慢放下一侧手臂和对侧腿，快接近地面时停住，再回到起始位置，换另一侧。",
                "你应该感觉腹部在控制身体，腰不要拱起来；如果拱腰，就把动作幅度变小。",
            ]
        if "毛巾划船" in exercise or exercise == "划船":
            return [
                "毛巾划船是居家背部训练动作，用毛巾作为拉拽支点，模拟划船训练背阔肌、中背和肩胛控制。",
                "基本做法：双手抓住固定好的毛巾，身体向后倾，脚踩稳，身体保持从头到脚一条直线。",
                "开始动作：先把肩胛骨向后向下收，再弯曲手肘把胸口拉向毛巾方向，然后慢慢回到起始位置。",
                "身体越接近水平，难度越高；新手先站得更直一点，降低难度。",
            ]
        if "臀桥" in exercise:
            return [
                "臀桥是主要训练臀部和髋伸展能力的动作，也常用于下肢和腰背友好训练。",
                "基本做法：仰卧屈膝，脚跟靠近臀部，双脚踩稳地面。",
                "开始动作：收紧臀部把髋部抬起，到肩、髋、膝接近一条线，再慢慢放下。",
                "重点是臀部发力，不是用腰往上顶。",
            ]
        if "超人式背伸" in exercise:
            return [
                "超人式背伸是一个俯卧位背部和臀部后侧链训练，主要练竖脊肌、臀部和肩胛后侧控制。",
                "基本做法：趴在垫子上，双腿伸直，脚背贴地或轻轻离地，双手可以向前伸，也可以放在身体两侧降低难度。",
                "开始动作：先轻轻收紧腹部和臀部，再把胸口、手臂和双腿小幅度抬离地面，不需要抬很高。",
                "在最高点停 1 秒，感受背部和臀部发力，然后慢慢放回地面；全程不要猛甩、不要憋气。",
                "新手可以先只抬上半身，或只抬对侧手脚，腰不舒服就停止。",
            ]
        return [
            f"{exercise} 是一个健身训练动作，主要作用取决于动作模式和发力部位。",
            "如果你不确定怎么做，先用低强度版本学习轨迹，保证动作可控、无痛。",
            "你可以告诉我是在家还是健身房练，我可以按你的器械条件拆成一步一步做法。",
        ]

    def _exercise_detail_tips(self, exercise: str) -> list[str]:
        if "毛巾划船" in exercise or exercise == "划船":
            return [
                "先确认毛巾固定点足够牢，不要挂在会滑动或松动的门把手上。",
                "身体保持一条直线，收紧腹部和臀部，避免用腰往后甩。",
                "先让肩胛骨向后向下收，再把手肘往身体两侧拉，不要耸肩。",
                "下放时慢一点，手臂伸直但肩不要被完全拽松；每组留 2-3 次余力。",
            ]
        if "深蹲" in exercise:
            return [
                "脚跟踩稳，膝盖方向大致跟脚尖一致，不要明显内扣。",
                "先用椅子或箱子控制深度，能稳定再逐步加深。",
                "全程保持躯干稳定，不要为了蹲深而塌腰或脚跟离地。",
            ]
        if "臀桥" in exercise:
            return [
                "脚跟靠近臀部，发力时想象用臀部把髋顶起来。",
                "顶端不要过度挺腰，肋骨保持收住。",
                "如果腰酸明显，降低幅度或先暂停。",
            ]
        if "死虫" in exercise or "鸟狗" in exercise or "侧桥" in exercise:
            return [
                "重点是腰背稳定，不追求速度和次数。",
                "动作过程中保持自然呼吸，不要憋气。",
                "如果腰痛增加，立刻停止并改做更小幅度版本。",
            ]
        if "超人式背伸" in exercise:
            return [
                "抬起幅度要小，重点是可控发力，不是把身体抬得越高越好。",
                "不要用惯性甩起上半身和腿，抬起和放下都要慢。",
                "如果腰部有挤压感、刺痛或疼痛增加，立刻停止，改成鸟狗或更小幅度背伸。",
                "做的时候保持自然呼吸，不要憋气硬撑。",
            ]
        return [
            "先用低强度版本学习动作轨迹，不急着加重量。",
            "动作过程保持可控，不借惯性完成。",
            "出现明显疼痛、麻木或刺痛时停止训练。",
        ]

    def _exercise_safety_fallback(self, exercise: str) -> str:
        if "毛巾划船" in exercise or "划船" in exercise:
            return "如果做的时候肩前侧、手腕或腰明显不舒服，先停下来，改成弹力带划船或坐姿划船。"
        if "死虫" in exercise:
            return "如果腰拱起来或腰痛增加，先缩小手脚下放幅度；还是不舒服就只做呼吸和骨盆控制练习。"
        if "超人式背伸" in exercise:
            return "如果腰部有挤压感或疼痛增加，先停下来，改做鸟狗或只抬上半身的小幅度版本。"
        if "深蹲" in exercise:
            return "如果膝盖或腰不舒服，先改成椅子辅助箱式深蹲，并减少下蹲深度。"
        if "臀桥" in exercise:
            return "如果腰酸明显，先降低抬髋高度，确认是臀部发力；仍不舒服就暂停。"
        return "如果出现明显疼痛、麻木或刺痛，先停止训练，必要时咨询医生或康复师。"

    def _exercise_profile(self, query: str, context: str) -> dict[str, Any]:
        return {
            "target": self._target_area(context),
            "setting": self._training_setting(query) or self._training_setting(context),
            "equipment": self._equipment(query) or self._equipment(context),
            "pain": self._pain_area(query),
            "level": self._training_level(query) or self._training_level(context),
        }

    def _training_goal_plan_answer(self, query: str, history: list[dict[str, str]]) -> str:
        goal = self._training_goal(query)
        context = (
            f"用户目标：{goal}\n"
            "请配置一个可执行但不过度复杂的健身计划。需要包含每周频率、训练结构、强度控制、"
            "饮食或恢复提醒，并在缺少训练经验、器械、伤病信息时先追问关键问题。"
        )
        generated = self.llm.answer(query, context, history, intent_code=2)
        if generated:
            return generated
        return (
            f"我判断你的锻炼目的更偏向：{goal}。\n\n"
            "DeepSeek 没有启用或调用失败，先给本地简版计划：\n"
            "1. 每周 3 次力量训练，优先全身训练，动作保留 2-3 次余力。\n"
            "2. 每周 2 次 20-30 分钟中低强度有氧。\n"
            "3. 每天保证蛋白质和睡眠，连续 2 周记录体重、围度和训练完成度。\n\n"
            "要让 DeepSeek 生成更完整计划，请在 `.env` 配置 `DEEPSEEK_API_KEY` 后重启服务。"
        )

    def _health_assessment_answer(self, query: str) -> str:
        metrics = self._extract_body_metrics(query)
        if not metrics:
            return "请按这个格式告诉我：身高 175cm，体重 80kg。我会计算 BMI，并用医学文献说明是否建议开始运动。"
        height_cm, weight_kg = metrics
        height_m = height_cm / 100
        bmi = weight_kg / (height_m * height_m)
        category = self._bmi_category(bmi)
        documents = self.vector_store.search("medical", "physical inactivity obesity BMI exercise health risk", limit=3)
        has_literature = bool([item for item in documents if item["score"] > 0] or self._fallback_medical_docs())
        evidence_note = "依据医学文献综合判断，" if has_literature else ""
        return (
            f"按你输入的身高体重估算：BMI = {bmi:.1f}，属于「{category}」。\n\n"
            f"我的判断：{self._fitness_need(category)}\n\n"
            f"{evidence_note}不运动的主要坏处通常包括："
            "心肺功能下降、脂肪和血糖血脂管理变差、肌肉量下降、腰背不适风险上升，"
            "以及长期慢病风险增加。这个判断不能替代医生诊断；如果有疾病或疼痛，请先咨询医生。"
        )

    def _target_area(self, query: str) -> str:
        lowered = query.lower()
        mapping = [
            (("背", "背部", "back"), "背部"),
            (("胸", "胸部"), "胸部"),
            (("腿", "下肢", "膝"), "下肢"),
            (("肩", "肩部"), "肩部"),
            (("臀", "臀部"), "臀部"),
            (("核心", "腹", "腰"), "核心/腰腹稳定"),
        ]
        for words, target in mapping:
            if any(word in lowered for word in words):
                return target
        return "全身基础体能"

    def _exercise_options(self, profile: dict[str, Any]) -> list[str]:
        target = profile["target"]
        setting = profile["setting"]
        equipment = profile["equipment"]
        pain = profile["pain"]
        level = profile["level"]
        if pain == "膝部":
            return ["臀桥：2-3 组，每组 10-12 次，先避开膝盖压力。", "坐姿腿屈伸等长收缩：每次保持 10-20 秒。", "低台阶上步：每侧 6-8 次，过程中膝盖不内扣，疼痛就停止。"]
        if pain == "腰部":
            return ["死虫：2 组，每组 6-8 次，腰不要拱起。", "鸟狗：2 组，每侧 6-8 次，骨盆保持稳定。", "臀桥：2 组，每组 10 次，腰部无痛再做。"]
        if "背部" in target:
            if equipment == "弹力带":
                return ["弹力带划船：3 组，每组 10-12 次。", "弹力带面拉：2-3 组，每组 12 次。", "俯身 Y-T-W：2 组，每个方向 8 次。"]
            if setting == "在家练":
                return ["弹力带划船或毛巾划船：2-3 组，每组 10-12 次。", "俯身 Y-T-W：2 组，每个方向 8-10 次。", "超人式背伸：2 组，每组 8-10 次，腰不舒服就跳过。"]
            return ["坐姿划船或弹力带划船：2-3 组，每组 10-12 次。", "高位下拉或辅助引体：2-3 组，每组 8-10 次。", "俯身哑铃划船：重量先轻，保持背部稳定。"]
        if "核心" in target:
            return ["死虫：2 组，每组 6-8 次，腰不要拱起。", "鸟狗：2 组，每侧 6-8 次，骨盆保持稳定。", "侧桥：2 组，每侧 15-25 秒。"]
        if "胸部" in target:
            if equipment == "哑铃":
                return ["哑铃卧推或地板卧推：2-3 组，每组 8-10 次。", "上斜俯卧撑：2 组，每组 8-12 次。", "哑铃飞鸟轻重量：2 组，每组 10 次，肩不适就取消。"]
            if setting == "在家练":
                return ["上斜俯卧撑：2-3 组，每组 8-12 次。", "跪姿俯卧撑：2 组，每组 6-10 次。", "弹力带推胸：动作慢一点，控制回程。"]
            return ["上斜俯卧撑：2-3 组，每组 8-12 次。", "器械胸推或哑铃卧推：先轻重量学习轨迹。", "弹力带夹胸：动作慢一点，感受胸部发力。"]
        if "下肢" in target:
            if level == "新手":
                return ["椅子辅助箱式深蹲：2 组，每组 8 次。", "臀桥：2 组，每组 10 次。", "扶墙提踵：2 组，每组 12 次。"]
            if equipment == "哑铃":
                return ["哑铃杯式深蹲：2-3 组，每组 8-10 次。", "哑铃罗马尼亚硬拉：2 组，每组 8 次。", "保加利亚分腿蹲退阶版：每侧 6-8 次。"]
            if setting == "在家练":
                return ["箱式深蹲：2-3 组，每组 8-10 次，可坐到椅子再站起。", "臀桥：2-3 组，每组 10-12 次。", "靠墙静蹲或台阶上步：2 组，膝盖保持稳定。"]
            return ["箱式深蹲：2-3 组，每组 8-10 次。", "臀桥：2-3 组，每组 10-12 次。", "台阶上步：每侧 8-10 次，膝盖保持稳定。"]
        return ["深蹲或箱式深蹲：2 组，每组 8-10 次。", "水平拉：弹力带划船或坐姿划船 2 组。", "核心稳定：死虫和侧桥各 2 组。"]

    def _training_setting(self, query: str) -> str | None:
        lowered = query.lower()
        if any(word in lowered for word in ("在家", "家练", "居家")):
            return "在家练"
        if "健身房" in lowered:
            return "健身房练"
        return None

    def _equipment(self, query: str) -> str | None:
        lowered = query.lower()
        if "弹力带" in lowered:
            return "弹力带"
        if "哑铃" in lowered:
            return "哑铃"
        if "杠铃" in lowered:
            return "杠铃"
        if any(word in lowered for word in ("徒手", "无器械", "没器械")):
            return "徒手"
        return None

    def _pain_area(self, query: str) -> str | None:
        lowered = query.lower()
        if "膝" in lowered and any(word in lowered for word in ("疼", "痛", "不好", "不舒服")):
            return "膝部"
        if "腰" in lowered and any(word in lowered for word in ("疼", "痛", "不好", "不舒服")):
            return "腰部"
        if "肩" in lowered and any(word in lowered for word in ("疼", "痛", "不好", "不舒服")):
            return "肩部"
        return None

    def _training_level(self, query: str) -> str | None:
        lowered = query.lower()
        if any(word in lowered for word in ("新手", "零基础", "刚开始")):
            return "新手"
        if any(word in lowered for word in ("练过", "有基础", "一年", "两年", "三年")):
            return "有基础"
        return None

    def _profile_sentence(self, profile: dict[str, Any]) -> str:
        parts = []
        for label in ("setting", "equipment", "pain", "level"):
            if profile.get(label):
                parts.append(profile[label])
        return f" 已结合条件：{'、'.join(parts)}。" if parts else ""

    def _training_goal(self, query: str) -> str:
        lowered = query.lower()
        if any(word in lowered for word in ("减脂", "减肥", "变瘦", "体脂")):
            return "减脂"
        if any(word in lowered for word in ("增肌", "变壮", "长肌肉")):
            return "增肌"
        if "塑形" in lowered or "马甲线" in lowered:
            return "塑形"
        return "综合体能提升"

    def _extract_body_metrics(self, query: str) -> tuple[float, float] | None:
        compact = query.replace("，", " ").replace(",", " ")
        height_match = re.search(r"(?:身高\s*)?(1[3-9]\d|2[0-2]\d)\s*(?:cm|厘米|公分)", compact, flags=re.I)
        if not height_match:
            height_match = re.search(r"身高\s*(1[3-9]\d|2[0-2]\d)", compact, flags=re.I)
        weight_match = re.search(r"体重\s*([4-9]\d|1[0-9]\d)\s*(kg|公斤|千克|斤)?", compact, flags=re.I)
        if not weight_match:
            weight_match = re.search(r"([4-9]\d|1[0-9]\d)\s*(kg|公斤|千克|斤)", compact, flags=re.I)
        if not height_match or not weight_match:
            return None
        height = float(height_match.group(1))
        weight = float(weight_match.group(1))
        unit = weight_match.group(2) if len(weight_match.groups()) >= 2 else ""
        if unit == "斤" or ("斤" in compact and weight > 90):
            weight = weight / 2
        return height, weight

    def _bmi_category(self, bmi: float) -> str:
        if bmi < 18.5:
            return "偏瘦"
        if bmi < 24:
            return "正常范围"
        if bmi < 28:
            return "超重"
        return "肥胖"

    def _fitness_need(self, category: str) -> str:
        if category == "正常范围":
            return "仍然建议规律运动，重点是维持心肺、肌肉量和腰背稳定，不一定以减重为目标。"
        if category == "偏瘦":
            return "建议做力量训练和营养管理，目标是提升肌肉量和基础体能，而不是继续减重。"
        return "建议开始规律运动，并结合饮食管理；先从低冲击有氧和基础力量训练开始更稳。"

    def _fallback_medical_docs(self) -> list[dict[str, Any]]:
        return [
            {
                "title": "WHO Guidelines on physical activity and sedentary behaviour",
                "content": "WHO 指南指出，成年人规律身体活动有助于降低心血管疾病、2 型糖尿病和部分癌症风险，并建议减少久坐。",
            },
            {
                "title": "Lee et al., The Lancet, 2012",
                "content": "该研究估算身体活动不足与多种非传染性疾病负担及预期寿命损失相关。",
            },
            {
                "title": "Ekelund et al., The Lancet, 2016",
                "content": "该荟萃分析显示，较高身体活动水平可减弱久坐时间与死亡风险之间的关联。",
            },
        ]
