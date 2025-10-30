# Copyright (C) 2025-present Naver Corporation. All rights reserved.

from transformers import AutoTokenizer, SiglipTextModel, CLIPTextModel, CLIPTokenizer, Siglip2TextModel
from torch import nn
import torch

MODEL_CONFIGS = {
    "siglip2": dict(
        hf_model="google/siglip2-base-patch16-224",
        embed_dim=768,
        tokenizer_args=dict(
            padding='max_length',
            max_length=64
        ),
        template="this is a photo of {}"
    ),
    "siglip": dict(
        hf_model="google/siglip-base-patch16-224",
        embed_dim=768,
        padding_mode='max_length',
        tokenizer_args=dict(padding='max_length'),
        template="This is a photo of {}."
    ),
    "clip": dict(
        hf_model="openai/clip-vit-base-patch32",
        embed_dim=512,
        tokenizer_args=dict(padding=True),
        template="a photo of {}"
    )
}


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, out_dim: int = 256, fixed_vocab: bool = False):
        assert model_name in MODEL_CONFIGS, f"Unknown text model: {model_name}"
        super().__init__()

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.embed_dim = self.config['embed_dim']

        self.change_mode(fixed_vocab=fixed_vocab)

    def change_mode(self, fixed_vocab: bool):
        self.fixed_vocab = fixed_vocab
        if not self.fixed_vocab:
            self.model, self.tokenizer = self.get_model()

    def get_model(self):
        if self.model_name == "siglip":
            tokenizer = AutoTokenizer.from_pretrained(self.config['hf_model'])
            model = SiglipTextModel.from_pretrained(self.config['hf_model'])
        elif self.model_name == "siglip2":
            tokenizer = AutoTokenizer.from_pretrained(self.config['hf_model'])
            model = Siglip2TextModel.from_pretrained(self.config['hf_model'])
        elif self.model_name == "clip":
            tokenizer = CLIPTokenizer.from_pretrained(self.config['hf_model'])
            model = CLIPTextModel.from_pretrained(self.config['hf_model'])
        else:
            raise ValueError(f"Unknown text model: {self.model_name}")
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        return model, tokenizer

    def embed_classes(self, model, tokenizer, classes, bs=32):
        text_in = [self.config['template'].format(c) for c in classes]
        embs = []
        with torch.no_grad():
            for i in range(0, len(text_in), bs):
                inputs = tokenizer(text_in[i:i + bs], return_tensors='pt', **self.config['tokenizer_args'])
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                lang_emb = model(**inputs).pooler_output.detach()
                embs.append(lang_emb)

        lang_emb = torch.cat(embs)

        return lang_emb

    def set_vocab(self, classes: list[str], device=None):
        model, tokenizer = self.get_model()
        if device is not None:
            model.to(device)

        lang_emb = self.embed_classes(model, tokenizer, classes)
        self.class_embeddings = {c: emb for c, emb in zip(classes, lang_emb)}

        # Free up memory
        del model, tokenizer
        torch.cuda.empty_cache()

    def forward(self, classes: list[str]):
        if self.fixed_vocab:
            assert all(c in self.class_embeddings for c in classes), "Missing classes in vocabulary. 'set_vocab' must be called if using fixed vocabulary"
            lang_emb = torch.stack([self.class_embeddings[c] for c in classes])
        else:
            lang_emb = self.embed_classes(self.model, self.tokenizer, classes)

        out = lang_emb / lang_emb.norm(dim=-1, keepdim=True)

        return out
