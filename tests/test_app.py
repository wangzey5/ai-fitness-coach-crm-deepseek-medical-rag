from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crm.api import create_app
from crm.config import Settings


class CRMAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        settings = Settings(Path(self.tempdir.name) / "test.db", None, "gpt-4.1-mini")
        self.app = create_app(settings)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_home_page(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        page = response.data.decode()
        self.assertIn("A3 健身私教 CRM", page)
        self.assertIn("开始对话", page)
        self.assertIn("手机尾号", page)
        self.assertIn("你对健身有什么问题", page)

        favicon = self.client.get("/favicon.ico")
        self.assertEqual(favicon.status_code, 204)

    def test_session_start_creates_named_profile(self) -> None:
        response = self.client.post("/api/session/start", json={"name": "王明", "phone_tail": "1234"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["user"]["name"], "王明")
        self.assertEqual(data["user"]["phone_tail"], "1234")
        self.assertFalse(data["is_new_user"] is None)

        chat = self.client.post(
            "/chat",
            json={"name": "王明", "phone_tail": "1234", "query": "我想练背"},
        )
        self.assertEqual(chat.status_code, 200)
        events = [json.loads(line) for line in chat.data.decode().splitlines()]
        self.assertEqual(events[-1]["data"]["user"]["name"], "王明")
        self.assertEqual(events[-1]["data"]["user"]["phone_tail"], "1234")

    def test_session_start_rejects_invalid_phone_tail(self) -> None:
        response = self.client.post("/api/session/start", json={"name": "王明", "phone_tail": "12"})
        self.assertEqual(response.status_code, 400)

    def test_vector_update_and_chat(self) -> None:
        update = self.client.post(
            "/api/vectorstores/workouts/documents",
            json={"title": "深蹲", "content": "可作为下肢力量训练的基础动作，适合在动作稳定后逐步加量。"},
        )
        self.assertEqual(update.status_code, 201)

        response = self.client.post("/chat", json={"query": "我想练背"})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.data.decode().splitlines()]
        self.assertEqual(events[0]["intent_code"], 1)
        self.assertEqual(events[-1]["type"], "done")
        self.assertIn("背部", events[-1]["data"]["response"])
        self.assertNotIn("搜索来源", events[-1]["data"]["response"])
        self.assertNotIn("https://", events[-1]["data"]["response"])

    def test_sql_agent_rejects_writes(self) -> None:
        response = self.client.post("/api/sql/query", json={"sql": "DELETE FROM users"})
        self.assertEqual(response.status_code, 400)

    def test_customer_summary(self) -> None:
        self.client.post(
            "/chat",
            json={"query": "我想减脂，一周能练 3 次，帮我做训练计划"},
        )
        response = self.client.post(
            "/api/sql/query",
            json={
                "sql": "SELECT phone_number, preferred_topic FROM users WHERE phone_number = ?",
                "parameters": ["local-customer"],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["rows"][0]["preferred_topic"], "training_goal_plan")

    def test_goal_plan_uses_deepseek_path_or_fallback(self) -> None:
        response = self.client.post("/chat", json={"query": "我想增肌，一周练三次"})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.data.decode().splitlines()]
        answer = events[-1]["data"]["response"]
        self.assertEqual(events[0]["intent_code"], 2)
        self.assertIn("增肌", answer)
        self.assertIn("DeepSeek", answer)

    def test_exercise_follow_up_keeps_previous_body_part(self) -> None:
        self.client.post("/chat", json={"query": "我要练腿"})
        response = self.client.post("/chat", json={"query": "在家练"})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.data.decode().splitlines()]
        answer = events[-1]["data"]["response"]
        self.assertEqual(events[0]["intent_code"], 1)
        self.assertIn("下肢", answer)
        self.assertIn("在家练", answer)
        self.assertIn("箱式深蹲", answer)
        self.assertNotIn("全身基础体能", answer)

    def test_exercise_answers_change_with_constraints(self) -> None:
        home = self.client.post("/chat", json={"query": "我想练腿，在家练"}).data.decode()
        with tempfile.TemporaryDirectory() as tempdir:
            knee_app = create_app(Settings(Path(tempdir) / "test.db", None, "gpt-4.1-mini"))
            knee = knee_app.test_client().post("/chat", json={"query": "我想练腿，但是膝盖不好"}).data.decode()
        with tempfile.TemporaryDirectory() as tempdir:
            dumbbell_app = create_app(Settings(Path(tempdir) / "test.db", None, "gpt-4.1-mini"))
            dumbbell = dumbbell_app.test_client().post("/chat", json={"query": "我想练腿，有哑铃"}).data.decode()

        home_answer = [json.loads(line) for line in home.splitlines()][-1]["data"]["response"]
        knee_answer = [json.loads(line) for line in knee.splitlines()][-1]["data"]["response"]
        dumbbell_answer = [json.loads(line) for line in dumbbell.splitlines()][-1]["data"]["response"]

        self.assertIn("箱式深蹲", home_answer)
        self.assertIn("膝部", knee_answer)
        self.assertIn("低台阶上步", knee_answer)
        self.assertIn("哑铃杯式深蹲", dumbbell_answer)
        self.assertNotEqual(home_answer, knee_answer)
        self.assertNotEqual(home_answer, dumbbell_answer)

    def test_specific_exercise_follow_up_answers_action_details(self) -> None:
        self.client.post("/chat", json={"query": "我想练背"})
        self.client.post("/chat", json={"query": "在家练"})
        response = self.client.post("/chat", json={"query": "毛巾划船有什么要注意的点吗"})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.data.decode().splitlines()]
        answer = events[-1]["data"]["response"]
        self.assertEqual(events[0]["intent_code"], 1)
        self.assertIn("毛巾划船", answer)
        self.assertIn("肩胛", answer)
        self.assertNotIn("搜索来源", answer)
        self.assertNotIn("https://", answer)
        self.assertNotIn("我可以帮你做三件事", answer)

    def test_exercise_explanation_differs_from_caution(self) -> None:
        explain = self.client.post("/chat", json={"query": "什么是死虫"})
        self.assertEqual(explain.status_code, 200)
        explain_answer = [json.loads(line) for line in explain.data.decode().splitlines()][-1]["data"]["response"]

        caution = self.client.post("/chat", json={"query": "死虫有什么注意点"})
        self.assertEqual(caution.status_code, 200)
        caution_answer = [json.loads(line) for line in caution.data.decode().splitlines()][-1]["data"]["response"]

        self.assertIn("核心稳定", explain_answer)
        self.assertIn("基本做法", explain_answer)
        self.assertIn("注意点", caution_answer)
        self.assertIn("腰背稳定", caution_answer)
        self.assertNotIn("坐姿划船", caution_answer)
        self.assertNotEqual(explain_answer, caution_answer)

    def test_superman_explanation_is_step_by_step(self) -> None:
        response = self.client.post("/chat", json={"query": "什么是超人式背伸"})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.data.decode().splitlines()]
        answer = events[-1]["data"]["response"]
        self.assertIn("趴在垫子上", answer)
        self.assertIn("小幅度抬离地面", answer)
        self.assertIn("慢慢放回地面", answer)
        self.assertNotIn("主要作用取决于动作模式", answer)

    def test_body_metrics_routes_to_medical_rag_assessment(self) -> None:
        self.client.post(
            "/api/vectorstores/medical/documents",
            json={
                "title": "WHO Guidelines",
                "content": "规律身体活动有助于降低心血管疾病、2 型糖尿病和部分癌症风险。",
            },
        )
        response = self.client.post("/chat", json={"query": "身高 175cm 体重 82kg，需要健身吗"})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.data.decode().splitlines()]
        answer = events[-1]["data"]["response"]
        self.assertEqual(events[0]["intent_code"], 3)
        self.assertIn("BMI", answer)
        self.assertIn("超重", answer)
        self.assertIn("医学文献", answer)
        self.assertNotIn("RAG", answer)
        self.assertNotIn("https://", answer)

    def test_local_intent_routes(self) -> None:
        service = self.app.extensions["crm"]["chat_service"]
        self.assertEqual(service.detect_intent_code("我想练背"), 1)
        self.assertEqual(service.detect_intent_code("我想减脂，一周训练三次，推荐一个计划"), 2)
        self.assertEqual(service.detect_intent_code("身高 175cm 体重 82kg"), 3)
        self.assertEqual(service.detect_intent_code("我腰不好"), 1)
        self.assertEqual(service.detect_intent_code("毛巾划船有什么要注意的吗"), 1)


if __name__ == "__main__":
    unittest.main()
