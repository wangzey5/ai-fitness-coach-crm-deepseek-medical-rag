from __future__ import annotations

from crm.config import Settings
from crm.database import Database
from crm.vector_store import VectorStore


DOCUMENTS = [
    (
        "needs",
        "健身目标分析清单",
        "推荐训练方案前，应确认主要目标是减脂、增肌、塑形、体能提升还是恢复训练；还要了解年龄、身高体重、训练经验、每周可训练次数、单次训练时长、可用器械、饮食作息、伤病史和身体限制。信息不足时应先追问，不应直接给出高强度计划。",
    ),
    (
        "needs",
        "训练目标匹配建议",
        "减脂客户优先建立稳定训练频率、日常活动量和饮食热量管理，可结合力量训练与中低强度有氧；增肌客户应重视渐进超负荷、动作质量、蛋白质摄入和睡眠恢复；新手塑形客户应先学习基础动作模式，避免一开始安排过多高强度间歇训练。",
    ),
    (
        "workouts",
        "新手一周三练示例",
        "新手可采用一周三次全身训练，每次包含热身、深蹲或腿举、髋铰链动作、水平推、水平拉、核心训练和放松。前 2 到 4 周以学习动作和建立习惯为主，强度控制在还能保留 2 到 3 次余力，不建议频繁冲击极限重量。",
    ),
    (
        "workouts",
        "课程与动作对比方法",
        "对比训练方式时，应结合目标、训练基础、关节承受能力、可用时间和恢复能力。跑步和椭圆机都可用于有氧，椭圆机冲击更低；HIIT 时间效率高但对心肺和恢复要求更高；力量训练更适合长期塑形和维持肌肉量。动作学习应优先保证稳定、可控和无痛。",
    ),
    (
        "medical",
        "WHO Guidelines on physical activity and sedentary behaviour",
        "世界卫生组织 2020 年身体活动和久坐行为指南指出，成年人规律身体活动有助于降低全因死亡、心血管疾病、2 型糖尿病和部分癌症风险，并建议减少久坐时间。来源：https://www.who.int/publications/i/item/9789240015128",
    ),
    (
        "medical",
        "Lee et al., The Lancet, 2012",
        "Lee 等人在 The Lancet 发表的身体活动不足疾病负担研究指出，身体活动不足与冠心病、2 型糖尿病、乳腺癌、结肠癌等非传染性疾病负担及预期寿命损失相关。来源：https://doi.org/10.1016/S0140-6736(12)61031-9",
    ),
    (
        "medical",
        "Ekelund et al., The Lancet, 2016",
        "Ekelund 等人在 The Lancet 发表的荟萃分析显示，较高身体活动水平可以减弱久坐时间与死亡风险之间的关联。来源：https://doi.org/10.1016/S0140-6736(16)30370-1",
    ),
    (
        "medical",
        "Warburton and Bredin, Current Opinion in Cardiology, 2017",
        "Warburton 和 Bredin 的综述总结了规律身体活动与心血管、代谢、肌肉骨骼和心理健康获益之间的关系，并强调活动不足是重要的可改变风险因素。来源：https://doi.org/10.1097/HCO.0000000000000437",
    ),
]


def main() -> None:
    settings = Settings.from_env()
    database = Database(settings.database_path)
    database.initialize()
    database.delete_seed_documents()
    store = VectorStore(database)
    for index, (collection, title, content) in enumerate(DOCUMENTS, start=1):
        store.upsert(
            collection,
            title,
            content,
            {"source": "seed", "industry": "fitness_coaching"},
            document_id=f"fitness-coaching-seed-{index}",
        )
    print(f"Seeded {len(DOCUMENTS)} documents into {settings.database_path}")


if __name__ == "__main__":
    main()
