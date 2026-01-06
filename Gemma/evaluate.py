import os
import math
import json
import torch
import gc
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba
from rouge_score import rouge_scorer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
# 新增：引入PeftModel用于加载LoRA权重
from peft import PeftModel
from data_preparation import load_custom_dataset, prepare_tokenized_dataset

# 下载必要的NLTK数据（若未下载过，取消注释）
# nltk.download('punkt_tab')


def clear_gpu_memory():
    """清理GPU显存"""
    torch.cuda.empty_cache()
    gc.collect()


def load_lora_model(base_model_path, lora_model_path):
    """
    加载“原始基座模型 + LoRA权重”（核心修改）
    :param base_model_path: 原始基座模型路径（如Qwen-7B、Llama-2-7B等，需与训练时一致）
    :param lora_model_path: 训练好的LoRA权重路径（即./lora_model文件夹）
    :return: 加载LoRA后的完整模型、tokenizer
    """
    # 1. 先加载原始基座模型的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right"  # 右padding，避免生成时警告
    )
    # 补充pad_token（若基座模型无pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 加载原始基座模型（与训练时的量化配置一致：8bit量化、BF16）
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动分配设备（CPU/GPU）
        trust_remote_code=True,
        load_in_8bit=True  # 保持与训练时一致的8bit量化，降低显存占用
    )

    # 3. 叠加LoRA权重（关键步骤：将训练好的LoRA适配层加载到基座模型）
    lora_model = PeftModel.from_pretrained(
        base_model,
        model_id=lora_model_path,
        device_map="auto"  # 与基座模型设备一致
    )

    # 4. 切换到评估模式（禁用Dropout，确保结果稳定）
    lora_model.eval()
    return lora_model, tokenizer


def compute_metrics(predictions, labels, tokenizer):
    """计算BLEU、ROUGE-L、损失和困惑度指标（逻辑不变，保留原有优化）"""
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_smoother = SmoothingFunction().method4

    decoded_preds = []
    decoded_labels = []

    # 1. 解码文本（NumPy argmax用axis，修复原错误）
    for pred_logits, label in zip(predictions, labels):
        pred_ids = pred_logits.argmax(axis=-1)  # NumPy数组用axis，而非PyTorch的dim
        # 解码预测文本
        pred_text = tokenizer.decode(
            pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        decoded_preds.append(pred_text)
        # 解码真实标签（过滤-100）
        label_filtered = [l for l in label if l != -100]
        label_text = tokenizer.decode(
            label_filtered,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        decoded_labels.append(label_text)

    # 2. 计算BLEU分数（中文jieba分词）
    bleu_scores = []
    for pred_text, label_text in zip(decoded_preds, decoded_labels):
        pred_tokens = jieba.lcut(pred_text.strip())
        label_tokens = [jieba.lcut(label_text.strip())]  # BLEU要求参考文本为列表的列表
        bleu = sentence_bleu(
            label_tokens,
            pred_tokens,
            smoothing_function=bleu_smoother,
            weights=(0.25, 0.25, 0.25, 0.25)  # 4-gram权重，平衡长短文本
        )
        bleu_scores.append(bleu)
    avg_bleu = np.mean(bleu_scores) * 100

    # 3. 计算ROUGE-L分数（关注语义结构匹配）
    rouge_scores = []
    for pred_text, label_text in zip(decoded_preds, decoded_labels):
        rouge_result = rouge_scorer_instance.score(label_text, pred_text)
        rouge_scores.append(rouge_result['rougeL'].fmeasure)
    avg_rouge_l = np.mean(rouge_scores) * 100

    # 4. 计算损失和困惑度（NumPy数组操作优化）
    # 拼接logits（predictions是list，每个元素为[seq_len, vocab_size]）
    predictions_np = np.concatenate(predictions, axis=0)
    # 拼接标签（过滤-100）
    labels_filtered = [label[label != -100] for label in labels]
    labels_np = np.concatenate(labels_filtered, axis=0)
    # 生成有效标签掩码（过滤-100）
    all_labels_flat = np.concatenate(labels, axis=0)
    mask = all_labels_flat != -100
    pred_flat_filtered = predictions_np[mask]

    # 计算交叉熵损失（转为PyTorch张量）
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(pred_flat_filtered, dtype=torch.float32),
        torch.tensor(labels_np, dtype=torch.long)
    ).item()
    # 计算困惑度（避免溢出）
    perplexity = math.exp(loss) if loss < 10 else float('inf')

    # 返回指标和样本结果
    return {
        "metrics": {
            "bleu_score": round(avg_bleu, 2),
            "rouge_l_score": round(avg_rouge_l, 2),
            "test_loss": round(loss, 4),
            "perplexity": round(perplexity, 2)
        },
        "samples": {
            "predictions_text": decoded_preds[:20],
            "labels_text": decoded_labels[:20]
        }
    }


def evaluate_on_test_set(model, tokenizer, test_dataset, output_dir):
    """在测试集上评估（逻辑不变，确保批次处理和内存清理）"""
    os.makedirs(output_dir, exist_ok=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("开始在测试集上进行评估...")
    # 初始化累加变量（避免存储全部logits，降低内存占用）
    total_bleu = 0.0
    total_rouge_l = 0.0
    total_loss = 0.0
    total_token = 0
    sample_count = 0
    decoded_preds = []
    decoded_labels = []

    # 小批次处理（根据显存调整，2为基础值，显存不足可改为1）
    batch_size = 2
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i + batch_size]
        # 准备输入（转为Tensor并分配到模型设备）
        inputs = {
            "input_ids": torch.tensor(batch["input_ids"]).to(model.device),
            "attention_mask": torch.tensor(batch["attention_mask"]).to(model.device),
            "labels": torch.tensor(batch["labels"]).to(model.device)
        }

        # 无梯度推理（评估阶段禁用梯度计算，节省显存）
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.to(dtype=torch.float32)  # BF16转FP32，避免NumPy不兼容
            batch_loss = outputs.loss.item()  # 直接用模型输出的损失（已自动过滤-100）

        # 1. 累加损失（按有效token数加权，确保平均损失准确）
        batch_token_count = (inputs["labels"] != -100).sum().item()
        total_loss += batch_loss * batch_token_count
        total_token += batch_token_count

        # 2. 解码文本并累加BLEU/ROUGE-L
        pred_ids = logits.argmax(axis=-1).cpu().numpy()  # 转NumPy数组用于解码
        label_ids = inputs["labels"].cpu().numpy()

        for pred_id, label_id in zip(pred_ids, label_ids):
            # 解码预测文本和真实标签
            pred_text = tokenizer.decode(pred_id, skip_special_tokens=True)
            label_text = tokenizer.decode([l for l in label_id if l != -100], skip_special_tokens=True)
            decoded_preds.append(pred_text)
            decoded_labels.append(label_text)  # 原代码笔误：此处应存label_text，非pred_text，已修正
            sample_count += 1

            # 计算单样本BLEU
            pred_tokens = jieba.lcut(pred_text.strip())
            label_tokens = [jieba.lcut(label_text.strip())]
            total_bleu += sentence_bleu(
                label_tokens,
                pred_tokens,
                smoothing_function=SmoothingFunction().method4
            )

            # 计算单样本ROUGE-L
            total_rouge_l += rouge_scorer.RougeScorer(
                ['rougeL'], use_stemmer=True
            ).score(label_text, pred_text)['rougeL'].fmeasure

        # 清理当前批次的显存
        clear_gpu_memory()
        # 打印进度
        print(f"已处理 {min(i + batch_size, len(test_dataset))}/{len(test_dataset)} 个样本")

    # 计算平均指标
    avg_bleu = (total_bleu / sample_count) * 100 if sample_count > 0 else 0.0
    avg_rouge_l = (total_rouge_l / sample_count) * 100 if sample_count > 0 else 0.0
    avg_loss = total_loss / total_token if total_token > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

    # 整理结果
    results = {
        "metrics": {
            "bleu_score": round(avg_bleu, 2),
            "rouge_l_score": round(avg_rouge_l, 2),
            "test_loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 2)
        },
        "samples": {
            "predictions_text": decoded_preds[:20],  # 仅保留前20条样本，避免结果文件过大
            "labels_text": decoded_labels[:20]
        }
    }

    # 保存结果到文件
    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results["metrics"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "test_samples.json"), "w", encoding="utf-8") as f:
        json.dump(results["samples"], f, ensure_ascii=False, indent=2)

    # 打印最终结果
    print("\n===== 测试集评估结果 =====")
    print(f"BLEU分数: {results['metrics']['bleu_score']}")
    print(f"ROUGE-L分数: {results['metrics']['rouge_l_score']}")
    print(f"测试损失: {results['metrics']['test_loss']}")
    print(f"困惑度: {results['metrics']['perplexity']}")
    print(f"\n结果已保存至 {output_dir}")
    return results
# 原始模型
def load_base_model(base_model_path):
    """只加载原始基座模型（不加载LoRA权重）"""
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True  # 可选：显存不足时启用
    )

    base_model.eval()
    return base_model, tokenizer


def main():
    # --------------------------
    # 核心配置：需根据你的实际路径修改！！！
    # --------------------------
    base_model_path = "."  # 原始基座模型路径（如Qwen-7B、Llama-2-7B，需与训练时一致）
    lora_model_path = "./data_r8_attention"  # 训练好的LoRA权重路径（即最终保存的lora_model文件夹）
    data_path = "dataset_with_think.jsonl"  # 测试用数据集路径（与训练时的数据集格式一致）
    output_dir = "./test/data_r8_attention"  # 测试结果保存目录（区分原output，避免覆盖）
    max_seq_length = 1024  # 最大序列长度（需与训练时一致，否则预处理不兼容）

    # 清理初始显存
    clear_gpu_memory()

    # 1. 加载数据集（包含train/val/test，后续仅用test集）
    print("加载数据集...")
    dataset = load_custom_dataset(data_path)

    # 🚫 不加载LoRA
    # print(f"加载原始模型: {base_model_path}")
    # model, tokenizer = load_base_model(base_model_path)

    # 2. 加载“基座模型 + LoRA权重”（核心修改步骤）
    print(f"加载基座模型: {base_model_path}")
    print(f"加载LoRA权重: {lora_model_path}")
    model, tokenizer = load_lora_model(base_model_path, lora_model_path)

    # 3. 预处理测试集（与训练时的预处理逻辑一致）
    print("预处理测试集...")
    tokenized_dataset = prepare_tokenized_dataset(
        dataset,
        tokenizer,
        max_seq_length
    )
    test_dataset = tokenized_dataset["test"]  # 提取测试集

    # 4. 在测试集上执行评估
    evaluate_on_test_set(model, tokenizer, test_dataset, output_dir)

    # 清理最终显存
    clear_gpu_memory()
    print("\n测试完成!")


if __name__ == "__main__":
    main()