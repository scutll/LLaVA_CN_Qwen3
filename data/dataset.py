from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, AutoProcessor
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import Dataset
from PIL import Image
import torch
import random
import io
import os
qwen_config_path = "../Qwen3-0.6B/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

base_path = os.path.dirname(os.path.abspath(__file__))
# Tokenizer
qwen_path = os.path.join(base_path, qwen_config_path)
dataset_name = "BelleGroup/multiturn_chat_0.8M"

# 训练中文闲聊能力并不需要图片，所以不需要CLIP的Processor
class LoRADataset_Multi(Dataset):
    def __init__(
        self,
        qwen_path,
        config=None,
        split_type='train',
        val_split_ratio=0.05,
    ):
        super().__init__()
        print(f"加载 {split_type} 数据集 ({dataset_name})")

        raw_dataset = load_dataset(dataset_name,
                                   split="train", 
                                   cache_dir=base_path,
                                   download_config=DownloadConfig(resume_download=True),
        )    
        
        split_dataset = raw_dataset.train_test_split(test_size=val_split_ratio, seed=520)
        # 这里有分train和test的数据集
        if split_type == "train":
            self.dataset = split_dataset["train"]
            print(f"训练集大小: {len(self.dataset)}")
        else:
            self.dataset = split_dataset["test"]
            print(f"验证集大小: {len(self.dataset)}")
            
        # 加载Qwen和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        self.pad_token_id = self.tokenizer.pad_token_id
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 数据集有instruction和output
        full_text = item["instruction"] + item["output"]
        # tokenization
        full_input_ids = self.tokenizer(full_text)["input_ids"]
        
        
        prompt_text = item["instruction"]
        
        # prompt长度,后面切割用
        prompt_len = len(self.tokenizer(prompt_text)["input_ids"])
        
        # labels是output的内容,前面用pad补齐长度
        labels = [self.pad_token_id] * prompt_len + full_input_ids[prompt_len:]
        
        #shift
        input_ids = full_input_ids[:-1]
        labels = labels[1:]
        
        
        return{
            "input_ids": input_ids,
            "labels": labels,
        }
        

class MTalkDataCollator(DataCollatorForSeq2Seq):
    
    def __call__(self, features, return_tensors=None):
        
        batch = super().__call__(features, return_tensors)
        
        return batch
        

if __name__ == "__main__":
    dataset = LoRADataset_Multi(qwen_path)
    item = random.choice(dataset)
    print(item['input_ids'])
    print(item['labels'])