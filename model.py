import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from path_config import qwen_config_path, clip16_config_path
from transformers.modeling_outputs import CausalLMOutputWithPast
import os

base_path = os.path.dirname(os.path.abspath(__file__))
qwen_path = os.path.join(base_path, qwen_config_path)
clip_path = os.path.join(base_path, clip16_config_path)


class VLMConfig(PretrainedConfig):
    model_type = "vlm"

    def __init__(
        self,
        # qwen语言模型
        qwen_path=qwen_path,
        # clip视觉模型
        clip_path=clip_path,
        # 图像patch的占位符数量
        # 图像patch就是将整张图像裁切成一小块一小块的区域，然后将小块当作“词”输入模型
        # patch32模式即每个patch32x32, 224x224被切成49个patch
        image_pad_num=49,
        # 冻结qwen参数
        qwen_frozen=True,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.qwen_path = qwen_path
        self.clip_path = clip_path
        self.image_pad_num = image_pad_num  # patch32 version: (224/32)**2 = 49
        self.qwen_frozen = qwen_frozen
        self.dtype = dtype



class VLM_without_CLIP(PreTrainedModel):
    
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        # 加载预训练的qwen和视觉模型
        # Qwen作为语言生成backbone
        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, dtype=config.dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_path)
        
        self.pad_id = self.tokenizer.pad_token_id

        if config.qwen_frozen:
            for param in self.qwen.parameters():
                param.requires_grad = False

    def forward(self, input_ids, labels, attention_mask=None):
        
        text_embeds = self.qwen.get_input_embeddings()(input_ids)
        
        qwen_outputs = self.qwen(
            inputs_embeds=text_embeds, attention_mask=attention_mask
        )
        
        logits = qwen_outputs.logits

        loss = None

        if labels is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_id)
            loss = loss_func(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1).to(logits.device),
            )
            
        return CausalLMOutputWithPast(loss=loss, logits=logits)




if __name__ == "__main__":
    config = VLMConfig() 
    vision_language_model = VLM_without_CLIP(config)
    params = sum(
        p.numel() for p in vision_language_model.parameters() if p.requires_grad
    )
    print(f"Trainable params: {params/1e6}M")
