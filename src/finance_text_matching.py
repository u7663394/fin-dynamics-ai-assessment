from rich import print
from transformers import AutoTokenizer, AutoModel

# 提供相似，不相似的语义匹配例子
examples = {
    '是': [
        ('公司ABC发布了季度财报，显示盈利增长。', '财报披露，公司ABC利润上升。'),
    ],
    '不是': [
        ('黄金价格下跌，投资者抛售。', '外汇市场交易额创下新高。'),
        ('央行降息，刺激经济增长。', '新能源技术的创新。')
    ]
}


def init_prompts():
    """
    初始化前置prompt，便于模型做 in-context learning。
    """
    pre_history = [
        (
            '现在你需要帮助我完成文本匹配任务，当我给你两个句子时，你需要回答我这两句话语义是否相似。只需要回答是否相似，不要做多余的回答。',
            '好的，我将只回答”是“或”不是“。'
        )
    ]
    for key, sentence_pairs in examples.items():
        # print(f'key--&gt;{key}')
        # print(f'sentence_pairs--&gt;{sentence_pairs}')
        for sentence_pair in sentence_pairs:
            sentence1, sentence2 = sentence_pair
            # print(f'sentence1--&gt;{sentence1}')
            # print(f'sentence2--&gt;{sentence2}')
            pre_history.append((f'句子一:{sentence1}\n句子二:{sentence2}\n上面两句话是相似的语义吗？',
                                key))
    return {"pre_history": pre_history}


def inference(
        sentence_pairs: list,
        custom_settings: dict
):
    """
    推理函数。

    Args:
        model (transformers.AutoModel): Language Model 模型。
        sentence_pairs (List[str]): 待推理的句子对。
        custom_settings (dict): 初始设定，包含人为给定的 few-shot example。
    """
    for sentence_pair in sentence_pairs:
        sentence1, sentence2 = sentence_pair
        sentence_with_prompt = f'句子一: {sentence1}\n句子二: {sentence2}\n上面两句话是相似的语义吗？'
        response, history = model.chat(tokenizer, sentence_with_prompt, history=custom_settings['pre_history'])
        print(f'&gt;&gt;&gt; [bold bright_red]sentence: {sentence_pair}')
        print(f'&gt;&gt;&gt; [bold bright_green]inference answer: {response}')
        # print(history)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(r"D:\github-repo\fin-dynamics-ai-assessment\weights\chatglm2-6b-int4",
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained(r"D:\github-repo\fin-dynamics-ai-assessment\weights\chatglm2-6b-int4",
                                      trust_remote_code=True).half().cuda()
    model = model.eval()

    sentence_pairs = [
        ('股票市场今日大涨，投资者乐观。', '持续上涨的市场让投资者感到满意。'),
        ('油价大幅下跌，能源公司面临挑战。', '未来智能城市的建设趋势愈发明显。'),
        ('利率上升，影响房地产市场。', '高利率对房地产有一定冲击。'),
    ]

    custom_settings = init_prompts()
    inference(
        sentence_pairs,
        custom_settings
    )
