import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from safetensors.torch import load_file as load_safetensors
from peft import get_peft_model, LoraConfig, TaskType
from path_config import qwen_config_path, clip16_config_path
from model import VLM, VLMConfig
import argparse

def generate_response(mode, k):
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_path, "stage2_loss1p8", "checkpoint-6400") 
    qwen_path = os.path.join(base_path, qwen_config_path)
    clip_path = os.path.join(base_path, clip16_config_path) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载模型配置...")
    config = VLMConfig(qwen_path=qwen_path, clip_path=clip_path, dtype=torch.bfloat16)
    
    print("加载基础模型结构...")
    model = VLM(config)

    # 加载LoRA微调好的模型
    print("配置并应用LoRA结构...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # 核心的是这个qwen
    # 这里还只是加载LoRA层
    model.qwen = get_peft_model(model.qwen, lora_config)

    print(f"从 {checkpoint_path} 的 model.safetensors 加载完整权重...")
    checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"在 {checkpoint_path} 中找不到权重文件 model.safetensors !")
    
    # 加载权重
    state_dict = load_safetensors(checkpoint_file)
    model.load_state_dict(state_dict, strict=False)
    
    print("合并LoRA权重以提升推理速度...")
    # 这个操作是把LoRA的低秩矩阵合并到base权重，内存占用更低但会修改原始权重而且不能直接继续进行LoRA微调训练了(adapter已经被卸载)
    model.qwen = model.qwen.merge_and_unload()

    print("将模型移动到设备...")
    model.to(device)
    model.eval()

    # 加载完模型才合并tokenizer，减少内存负担
    print("加载分词器和图像处理器...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(clip_path)
    print("模型加载完成，进入多轮对话模式。")

    # 多轮对话的记忆
    conversation = [{"role": "system", "content": "You are a helpful assistant."}]
    pixel_values = None
    
    Mode = mode
    
    while True:
        try:
            prompt_input = input("用户: ")
            if prompt_input.lower() == 'exit':
                print("程序退出。")
                break
            if not prompt_input.strip():
                print("问题不能为空。")
                continue
            
            # 插入图片
            if pixel_values is None:
                image_path_input = input("图片路径 (可选, 直接按Enter跳过): ")
                if image_path_input.strip():
                    image_path = os.path.join(base_path, image_path_input)
                    if not os.path.exists(image_path):
                        print("图片路径不存在，请重试。")
                        continue
                    print("处理图片中...")
                    image = Image.open(image_path).convert("RGB")
                    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
                        device, dtype=model.config.dtype
                    )
                    if "<image>" not in prompt_input:
                        prompt_input += " <image>"
                else:
                    print("未提供图片。")

            template_prompt = prompt_input.replace("<image>", "<|image_pad|>" * model.image_pad_num)
            
            conversation.append({"role": "user", "content": template_prompt})
            
            formatted_prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(device)
            # initial_prompt_len = input_ids.shape[1]

            print("模型正在生成...")
            initial_prompt_len = input_ids.shape[1]
            max_new_tokens = 256  
            eos_token_id = tokenizer.eos_token_id
            
            # 模式选择 
            # Greedy: 选择概率最大的token
            # Top-k Sampling: 在top-k 候选集合中随机选择
            # Hybrid-Smart 在top-k中随机选择，当<eos>概率最大时直接选择eos
            print(f"token生成模式: {Mode}")    
            
            # 生成过程依然是每次一个token
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if pixel_values is not None:
                        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=None)
                    elif pixel_values is None:
                        outputs = model(input_ids=input_ids,
                                        labels=None)
                        
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Greedy
                    if Mode == "Greedy":
                        next_token_id = torch.argmax(next_token_logits, dim=-1)
                        if next_token_id.item() == eos_token_id:
                            print("(检测到结束符)")
                            break
                        else: 
                            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
                            continue
                    
                    # Top-k
                    if Mode == "TopkS":
                        # 使用top20来随机选择token
                        probs = torch.softmax(next_token_logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
                        
                        next_token_id = topk_indices[0, torch.multinomial(topk_probs[0], num_samples=1)]
                        next_token_id = next_token_id.unsqueeze(0)
                        
                        if next_token_id.item() == eos_token_id:
                            print("(检测到结束符)")
                            break
                        input_ids = torch.cat([input_ids, next_token_id], dim=1)
                        continue
                    
                    # Hybird
                    if Mode == "Hybrid":                   
                        next_token_id = torch.argmax(next_token_logits, dim=-1)
                        if next_token_id == eos_token_id:
                            print("检测到结束符优先")
                            break
                        probs = torch.softmax(next_token_logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
                        
                        next_token_id = topk_indices[0, torch.multinomial(topk_probs[0], num_samples=1)]
                        next_token_id = next_token_id.unsqueeze(0)
                        
                        input_ids = torch.cat([input_ids, next_token_id], dim=1)
                        
                        continue
                    
                    else:
                        print("Error: Unknown token generating Mode")
                        break

            response_ids = input_ids[:, initial_prompt_len:]
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            print("-" * 30)
            print("模型回答:", response_text)
            print("-" * 30 + "\n")

            # 这里是最重要的一部分：每次对话完成都将本轮对话加入到对话历史中并在下一次对话中将对话记录喂给ai，这样ai就能知道之前的对话  
            conversation.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\n程序被手动中断。")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("--Mode", type=str, choices=["Greedy", "TopkS", "Hybrid"], default="Hybrid", help="token generater mode -Greedy -Top-k_Sampling -Hybrid(default)")
    parser.add_argument("--k", type=int, default=20, help="top-k randomly selection")
    
    args = parser.parse_args()

    generate_response(mode=args.Mode, k=args.k)