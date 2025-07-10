from transformers import PretrainedConfig, PreTrainedModel, BertModel, BertConfig
import timm
import torch.nn as nn
import torch
import numpy
from torchvision import transforms
from PIL import Image

class KEEPConfig(PretrainedConfig):
    model_type = "keep"  # 标记模型类型

    def __init__(
        self,
        vision_config=None,  # Vision Encoder 的配置
        text_config=None,    # Text Encoder 的配置
        projection_dim=768,  # 投影维度，默认为 768
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.text_config = text_config
        self.projection_dim = projection_dim
        

class KEEPModel(PreTrainedModel):
    config_class = KEEPConfig  # 绑定到自定义配置类

    def __init__(self, config):
        super().__init__(config)

        # Vision Encoder (基于 timm 的 ViT)
        self.visual = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        # 线性投影层，将 Vision Encoder 的输出投影到 768 维
        self.visual_head = nn.Sequential(
                    nn.Linear(self.visual.num_features, config.projection_dim),
                    nn.GELU(),
                    nn.Linear(config.projection_dim, config.projection_dim)
                )

        # Text Encoder (基于 PubMedBERT)
        text_config =  BertConfig(**config.text_config)
        self.text = BertModel(text_config)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * numpy.log(1 / 0.04))

    def encode_image(self, image_inputs):
        vision_features = self.visual(image_inputs)  # [batch_size, vision_dim]
        vision_features =  torch.nn.functional.normalize(self.visual_head(vision_features), dim=-1)  # [batch_size, projection_dim]
        
        return vision_features
    
    def encode_text(self, text_inputs):
        text_features = torch.nn.functional.normalize(self.text(**text_inputs).pooler_output, dim=-1)  # [batch_size, text_dim]
        return text_features
    
    
    def forward(self, image_inputs, text_inputs):
        vision_features = self.encode_image(image_inputs)
        
        text_features = self.encode_text(text_inputs)

        # 返回两个独立的特征
        return {
            "vision_features": vision_features,  # 视觉特征
            "text_features": text_features       # 文本特征
        }