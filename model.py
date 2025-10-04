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
                
        # 加上CLIP模块使得模型可以加载stage1的权重，但再forward中不使用
        self.clip = AutoModel.from_pretrained(
            config.clip_path, dtype=config.dtype
        )
        self.dense1 = nn.Linear(
            self.clip.config.vision_config.hidden_size * 4, 
            self.qwen.config.hidden_size,
            dtype=config.dtype,
        )
        self.dense2 = nn.Linear(
            self.qwen.config.hidden_size,
            self.qwen.config.hidden_size,
            dtype=config.dtype,
        )
        for param in self.clip.parameters():
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


class VLM(PreTrainedModel):
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.qwen = AutoModelForCausalLM.from_pretrained(
            config.qwen_path, dtype=config.dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_path)
        self.clip = AutoModel.from_pretrained(
            config.clip_path, dtype=config.dtype
        )

        self.dense1 = nn.Linear(
            # 在clip-patch16中, hidden_size = 768, 后面要将4个patch拼在一起，所以是*4
            self.clip.config.vision_config.hidden_size * 4, # patch16 version
            # Qwen的embedding维度
            self.qwen.config.hidden_size,
            dtype=config.dtype,
        )
        # dense1之后的dense2是不改变形状而进行特征继续细化
        self.dense2 = nn.Linear(
            self.qwen.config.hidden_size,
            self.qwen.config.hidden_size,
            dtype=config.dtype,
        )
        
        self.image_pad_num = config.image_pad_num
        
        # 文本padding
        self.pad_id=self.tokenizer.pad_token_id
        # 图像专用占位符，在 forward 时会被替换为图像投影后的 embedding
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # 冻结参数
        for param in self.clip.parameters():
            param.requires_grad = False

        if config.qwen_frozen:
            for param in self.qwen.parameters():
                param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values = None, attention_mask=None):
        
        # 输入input_ids转成embedding
        text_embeds = self.qwen.get_input_embeddings()(input_ids)

        if pixel_values is not None:

            image_embeds = self.clip.vision_model(pixel_values).last_hidden_state[
                :, 1:, :
            ]  # 去掉cls token
            
            # (batch_size, patch数, 每个patch被编码成的向量维度)
            b, s, d = image_embeds.shape    # 如果使用patch16，可以解除这两行注释
            image_embeds = image_embeds.view(b, -1, 4 * d)  # (b, 49, 768 * 4) 
    
            # 进入线性层提取特征
            image_features = self.dense2(F.silu(self.dense1(image_embeds)))
            text_embeds = text_embeds.to(image_features.dtype)  #数据类型一致
            # 合并文本信息和图像信息
            text_embeds = self.merge_text_and_image(input_ids, text_embeds, image_features)
        
        qwen_outputs = self.qwen(
            inputs_embeds=text_embeds, attention_mask=attention_mask
        )
        
        # logits就是模型的输出(对每一个词的预测向量) (batch_size, seq_len, vocab_size)
        logits = qwen_outputs.logits

        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_id)
            loss = loss_func(
                # 展平为(b*l, vocab_size) 经过softmax层再进行交叉熵计算
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1).to(logits.device),
            )

        # 返回loss和logits
        return CausalLMOutputWithPast(loss=loss, logits=logits)




if __name__ == "__main__":
    config = VLMConfig() 
    vision_language_model = VLM_without_CLIP(config)
    params = sum(
        p.numel() for p in vision_language_model.parameters() if p.requires_grad
    )
    print(f"Trainable params: {params/1e6}M")
