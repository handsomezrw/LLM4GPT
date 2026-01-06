import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def load_custom_dataset(file_path="dataset_with_think.jsonl", test_size=0.15, val_size=0.15):
    """
    加载自定义JSONL数据集并划分为训练集、验证集和测试集

    Args:
        file_path: JSONL文件路径
        test_size: 测试集占总数据的比例
        val_size: 验证集占总数据的比例

    Returns:
        DatasetDict: 包含训练集、验证集和测试集的数据集字典
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))

        # 转换为列表格式进行划分
        dataset_list = data  # 直接使用列表更高效，无需先转Dataset

        # 第一次拆分：训练集 和 临时集（验证集+测试集）
        # 计算临时集占比：验证集比例 + 测试集比例
        temp_size = test_size + val_size
        train_data, temp_data = train_test_split(
            dataset_list,
            test_size=temp_size,
            random_state=42
        )

        # 第二次拆分：从临时集中拆分出验证集和测试集
        # 计算验证集在临时集中的占比（相对于临时集总量）
        val_ratio = val_size / temp_size
        val_data, test_data = train_test_split(
            temp_data,
            test_size=1 - val_ratio,  # 测试集占临时集的比例 = 1 - 验证集占临时集比例
            random_state=42
        )

        return DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        raise


def preprocess_function(examples, tokenizer, max_seq_length=1024):
    """
    修正版：使用 apply_chat_template 自动适配 Qwen 和 Llama 的格式
    """
    texts = []

    for q, t, a in zip(examples["question"], examples["think"], examples["answer"]):
        # 1. 构建标准消息结构
        # 我们把 '思考' 和 '回答' 都放在 Assistant 的内容里
        # 你可以自定义思考的显示格式，比如加上 <think> 标签，方便后续观察
        full_response = f"【思考过程】\n{t}\n\n【最终回答】\n{a}"

        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": full_response}
        ]

        # 2. 使用 tokenizer 自动应用模板
        # tokenize=False 表示先只转换成带特殊token的字符串，方便我们在后面统一tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)

    # 3. 统一 Tokenize
    # Llama 3 注意：如果 tokenizer 没有 pad_token，这里会报错，需要在外部设置
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",  # 建议：如果在DataCollator中动态padding，这里可以设为False以节省空间
        return_tensors="pt"
    )

    # 4. 生成 Labels
    # 注意：这里简单的 clone 会导致模型同时也学习“用户的问题”。
    # 严格的微调通常会将用户问题的部分在 labels 中设为 -100 (mask掉)。
    # 但为了保持和你原逻辑一致，这里暂时保留全量学习。
    tokenized["labels"] = tokenized["input_ids"].clone()

    # 将 padding 部分的标签设为 -100，避免模型学习 pad token
    tokenized["labels"][tokenized["input_ids"] == tokenizer.pad_token_id] = -100

    return tokenized


# --- 必须添加的外部调用逻辑 ---
# 在调用 preprocess_function 之前，初始化 tokenizer 时必须执行以下操作：
# tokenizer = AutoTokenizer.from_pretrained("你的Llama模型路径")
#
# # 【关键】Llama 3 必须要修补 pad_token，否则 padding="max_length" 会报错
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id


def prepare_tokenized_dataset(dataset, tokenizer, max_seq_length=1024):
    """
    对数据集进行预处理并返回tokenized后的数据集
    不需要大改，基本沿用你原来的逻辑即可
    """

    # 获取需要移除的列名。
    # 建议直接从 dataset 的第一个 split 中动态获取，防止硬编码 "train" 导致报错（虽然你的代码里肯定有 train）
    column_names = list(dataset.values())[0].column_names

    return dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_seq_length),
        batched=True,
        remove_columns=column_names,  # 移除原始文本列（question, think, answer），只保留 input_ids 和 labels
        num_proc=1,  # 如果数据量特别大，可以设为 4 或 8 开启多进程，但要注意 tokenizer 并发问题
        desc="Running tokenizer on dataset"  # 添加进度条描述
    )

