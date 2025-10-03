import torch
from data.dataset import LoRADataset_Multi, MTalkDataCollator
from transformers import Trainer, TrainingArguments, AutoTokenizer
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os
from path_config import qwen_config_path, clip16_config_path
from model import VLM_without_CLIP, VLMConfig

# 使用更多显存的时候减少发生OOM(out of memory)的概率
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

base_path = os.path.dirname(os.path.abspath(__file__))
qwen_path = os.path.join(base_path, qwen_config_path)
clip_path = os.path.join(base_path, clip16_config_path)
output_dir = os.path.join(base_path, "output_lora")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 预训练的权重经过单轮对话训练(stage1)后得到的权重
# 这里直接进行stage2
stage1_model_path = os.path.join(base_path, "stage1_patch16_loss3p9", "pretrained_best_model")


if __name__ == "__main__":

    # 训练数据类型: bf16
    TRAINING_TYPE = torch.bfloat16
    print("加载模型和分词器...")
    config = VLMConfig(qwen_path=qwen_path, clip_path=clip_path, dtype=TRAINING_TYPE)
    model = VLM_without_CLIP(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)

    # 导入stage1的模型
    checkpoint_file = os.path.join(stage1_model_path, "model.safetensors")
    if os.path.exists(checkpoint_file):
        print(f"从 {checkpoint_file} 加载第一阶段权重...")
        state_dict = load_safetensors(checkpoint_file)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"在 {stage1_model_path} 中找不到第一阶段的权重文件!")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数量（应用lora前）为：{params/1e6:.2f}M")


    # 配置LoRA开始stage2的训练
    print("配置并应用LoRA...")
    # 重要：LoRA配置的超参数
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    

    model.qwen = get_peft_model(model.qwen, lora_config)
    
    print("检查base权重冻结：")
    for name, p in model.qwen.named_parameters():
        print(name, p.requires_grad)
        
    print("检查 CLIP 权重冻结：")
    for name, p in model.clip.named_parameters():
        print(name, p.requires_grad)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数量（应用lora后）为：{params/1e6:.2f}M")

    # 构建数据集
    print("数据集构建...")
    train_dataset = LoRADataset_Multi(qwen_path, config, split_type="train")
    eval_dataset = LoRADataset_Multi(qwen_path, config, split_type="val")

    # DataCollator可以套用VLM使用的,因为数据形式其实完全一样
    data_collator = MTalkDataCollator(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )

    print("配置参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=0.3,
        weight_decay=0.05,
        warmup_ratio=0.05,
        optim="adamw_torch",  # adamw_8bit备用
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=200,
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        save_steps=200,
        eval_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_num_workers=24,
        dataloader_pin_memory=True,
    )

    print("实例化Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("开始训练...")
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    print("训练完成，正在保存最佳模型...")
    trainer.save_model(os.path.join(output_dir, "lora_best_model"))