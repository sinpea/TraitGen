import torch.nn as nn
import torch

import open_clip
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from src.config import LLM_ID,CLIP_ID,STEERING_PROMPT

class LLaVAProjector(nn.Module):
    """2-Layer MLP projecting BioCLIP patch tokens into the LLM embedding space."""
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            # nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class LLaVACUBModel(nn.Module):
    def __init__(self, llm_id=LLM_ID):
        super().__init__()
        # 1. BioCLIP-2 Vision Backbone (Frozen)
        
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_ID)
        
        for p in self.clip.parameters():
            p.requires_grad = False

        # Extract visual dimension
        # vision_dim = self.clip.visual.output_dim if hasattr(self.clip.visual, "output_dim") else 1024
        # Extract spatial patch token dimension (1024) instead of pooled projection dim
        
        if hasattr(self.clip.visual, "width"):
            vision_dim = self.clip.visual.width
        elif hasattr(self.clip.visual, "transformer"):
            vision_dim = self.clip.visual.transformer.width
        else:
            vision_dim = 1024  # Default fallback for BioCLIP-2 unpooled patches
        
        # 2. Causal LLM Decoder + LoRA
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map=None,
            trust_remote_code=True
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(base_llm, peft_config)

        # 3. LLaVA MLP Projection Bridge
        llm_dim = base_llm.config.hidden_size
        self.projector = LLaVAProjector(vision_dim, llm_dim)

    def extract_patch_tokens(self, pixel_values):
        """Extracts spatial image patches (196 tokens for 224x224 input)."""
        visual = self.clip.visual
        with torch.no_grad():
            if hasattr(visual, "forward_features"):
                res = visual.forward_features(pixel_values)
                if isinstance(res, dict) and "x_norm_patchtokens" in res:
                    return res["x_norm_patchtokens"].float()
                elif isinstance(res, torch.Tensor) and res.ndim == 3:
                    return res[:, 1:, :].float()

            # Fallback manual extraction for OpenCLIP ViT architectures
            x = visual.conv1(pixel_values)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            class_embed = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
            x = torch.cat([class_embed, x], dim=1) + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x).permute(1, 0, 2)
            x = visual.transformer(x).permute(1, 0, 2)
            patch_tokens = visual.ln_post(x)[:, 1:, :].float()
            return patch_tokens

    def forward(self, pixel_values, prompt_ids, prompt_mask, target_ids, target_mask):
        B = pixel_values.size(0)

        # Extract 196 patch tokens per image
        patch_tokens = self.extract_patch_tokens(pixel_values) # [B, 196, vision_dim]
        
        visual_embeds = self.projector(patch_tokens)
        # [B, 196, llm_dim]

        if torch.isnan(visual_embeds).any():
            print("None Values encountered")
        num_patches = visual_embeds.size(1)

        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        target_embeds = self.llm.get_input_embeddings()(target_ids)

        inputs_embeds = torch.cat([visual_embeds, prompt_embeds, target_embeds], dim=1)

        # Build Labels (-100 ignored in CrossEntropy)
        patch_labels = torch.full((B, num_patches), -100, device=pixel_values.device, dtype=torch.long)
        prompt_labels = torch.full(prompt_ids.shape, -100, device=pixel_values.device, dtype=torch.long)
        labels = torch.cat([patch_labels, prompt_labels, target_ids], dim=1)

        # Build Attention Mask
        patch_mask = torch.ones((B, num_patches), device=pixel_values.device, dtype=torch.long)
        full_mask = torch.cat([patch_mask, prompt_mask, target_mask], dim=1)

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask)
        logits = outputs.logits

        # Shift logits for causal language modeling
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)).view(shift_labels.size())

        # Mask padding
        weights = torch.ones_like(shift_labels, dtype=torch.float)
        weights[shift_labels == -100] = 0.0

        conclusion_ids = self.tokenizer("Deduction:", add_special_tokens=False).input_ids
        
        for b in range(B):
            tokens = shift_labels[b]
            for i in range(len(tokens) - len(conclusion_ids)):
                if torch.equal(tokens[i:i+len(conclusion_ids)], torch.tensor(conclusion_ids, device=tokens.device)):
                    weights[b, i:] *= 4.0  # Scale up attention on the final inference result
                    break        
        
        loss = (token_loss * weights).sum() / torch.clamp(weights.sum(), min=1.0)
        return loss

    @torch.no_grad()
    def generate(self, pixel_values, max_new_tokens=128):
        B = pixel_values.size(0)
        patch_tokens = self.extract_patch_tokens(pixel_values)
        visual_embeds = self.projector(patch_tokens)
        if torch.isnan(visual_embeds).any():
            print("None values encountered")
        num_patches = visual_embeds.size(1)

        prompt = self.tokenizer([STEERING_PROMPT] * B, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt.input_ids.to(pixel_values.device)
        prompt_mask = prompt.attention_mask.to(pixel_values.device)

        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)

        patch_mask = torch.ones((B, num_patches), device=pixel_values.device, dtype=torch.long)
        attention_mask = torch.cat([patch_mask, prompt_mask], dim=1)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=2,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.replace(STEERING_PROMPT, "").strip() for t in texts]
