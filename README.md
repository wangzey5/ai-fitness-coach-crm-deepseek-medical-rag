# A3 健身私教 CRM

一个面向健身房和私教工作室、可本地运行的 AI CRM。系统通过对话识别客户意图，分别调用搜索、DeepSeek 和医学 RAG，完成动作推荐、目标训练计划和身高体重健康评估。

系统包含四类意图：

1. 动作搜索推荐：判断用户需要练什么，联网搜索并标注来源网站
2. 锻炼目标与计划配置：判断减脂、增肌、塑形等目的，调用 DeepSeek 配置计划
3. 身高体重健康评估：根据身高体重计算 BMI，检索医学文献说明是否建议运动
4. 其他健身咨询

未配置模型时使用本地规则和向量检索；配置 DeepSeek 后，意图 2 会使用 DeepSeek 生成更完整的训练计划。意图 1 会调用搜索模块并展示来源网站，意图 3 会调用本地医学 RAG 文献库。

优先使用 DeepSeek：

```bash
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

也可以继续使用 OpenAI：

```bash
OPENAI_API_KEY=你的密钥
OPENAI_MODEL=gpt-4.1-mini
```

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m scripts.seed
python app.py
```

首次运行或更新示例健身资料后执行 `python -m scripts.seed`。示例训练建议仅用于一般咨询；如有疼痛、疾病、伤病史或特殊人群需求，应先咨询医生或现场专业教练。

## 接口

健康检查：

```bash
curl http://127.0.0.1:5000/health
```

更新本地向量库：

```bash
curl -X POST http://127.0.0.1:5000/api/vectorstores/workouts/documents \
  -H 'Content-Type: application/json' \
  -d '{"title":"训练或课程名称","content":"适用目标、训练安排和注意事项","metadata":{"source":"coach-notes","updated_at":"YYYY-MM-DD"}}'
```

建立或加载用户档案：

```bash
curl -X POST http://127.0.0.1:5000/api/session/start \
  -H 'Content-Type: application/json' \
  -d '{"name":"王明","phone_tail":"1234"}'
```

聊天接口返回 NDJSON 流：

```bash
curl -N -X POST http://127.0.0.1:5000/chat \
  -H 'Content-Type: application/json' \
  -d '{"name":"王明","phone_tail":"1234","query":"身高 175cm 体重 82kg，需要健身吗？"}'
```

结束前端会话：

```bash
curl -X POST http://127.0.0.1:5000/api/session/end
```

只读 SQL Agent：

```bash
curl -X POST http://127.0.0.1:5000/api/sql/query \
  -H 'Content-Type: application/json' \
  -d '{"sql":"SELECT lifecycle_stage, preferred_topic FROM users LIMIT 10"}'
```

## 测试

```bash
python -m unittest discover -s tests -v
```
