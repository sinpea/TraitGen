import re
import json

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

from src.datasets import CUBDataset
from src.model import LLaVACUBModel

from src.config import DEVICE,NUM_GPUS,JSON_PATH,JSON_PATH_TEST,EPOCHS,BATCH_SIZE,IMAGE_ROOT,CKPT_PATH
from src.config import METRICS_PATH,PLOT_PATH,TRAIN_SIZE,LR,WARMUP_RATIO

class MetricsTracker:
    def __init__(self):
        self.history = {"step_loss": [], "epoch_loss": [], "lr": [], "eval_acc": []}

    def log_step(self, loss, lr):
        self.history["step_loss"].append(loss)
        self.history["lr"].append(lr)

    def log_epoch(self, epoch_loss, eval_acc):
        self.history["epoch_loss"].append(epoch_loss)
        self.history["eval_acc"].append(eval_acc)

    def save(self, filepath=METRICS_PATH):
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=4)

    def plot_and_save(self, save_path=PLOT_PATH):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Training Loss per Step
        axes[0].plot(self.history["step_loss"], color="#2b5c8f", alpha=0.6, label="Step Loss")
        axes[0].set_title("Training Loss Over Steps")
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 2. Learning Rate Schedule
        axes[1].plot(self.history["lr"], color="#e05d06")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("LR")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # 3. Epoch Loss & Evaluation Accuracy
        if len(self.history["epoch_loss"]) > 0:
            epochs = range(1, len(self.history["epoch_loss"]) + 1)
            ax3_2 = axes[2].twinx()
            axes[2].plot(epochs, self.history["epoch_loss"], "b-o", label="Epoch Loss")
            ax3_2.plot(epochs, self.history["eval_acc"], "g-s", label="Accuracy")
            axes[2].set_xlabel("Epochs")
            axes[2].set_ylabel("Loss", color="b")
            ax3_2.set_ylabel("Accuracy", color="g")
            axes[2].set_title("Epoch Loss vs Species Accuracy")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📊 Training curves saved successfully to '{save_path}'")



@torch.no_grad()
def evaluate_accuracy(model, eval_loader, device,max_lim=30):
    def extract_species(text):
        if text is None:
            return None
    
        # Matches "Deduction: ", optional spaces, optional "a "/"an ", 
        # and captures everything up to a semicolon, period, or end of string.
        match = re.search(r"deduction:\s*(?:an?\s+)?([^;.]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
        return None
    
    model.eval()
    correct = 0
    total = 0
    with torch.amp.autocast("cuda"):
        # Unwrapped model reference for generate()
        raw_model = model.module if hasattr(model, "module") else model
    
        for batch in eval_loader:
            pv, prompt_ids, prompt_mask, target_ids, target_mask, true_captions = batch
            pv = pv.to(device)
            # 1. Autoregressively generate output string
            gen_outputs = raw_model.generate(
                pixel_values=pv,
                max_new_tokens=128
            )
            # Ensure gen_outputs is iterable if single string returned
            if isinstance(gen_outputs, str):
                gen_outputs = [gen_outputs]

            # 2. Evaluate predictions against ground truth
            for gen_text, true_caption in zip(gen_outputs, true_captions):
                # true_species = true_caption.split("Deduction:")[-1].replace(";", "").strip().lower()
                true_species = extract_species(true_caption)
                
                print("gentext:")
                print(gen_text)
                print("true caption:")
                print(true_caption)
                
                if true_species in gen_text.lower():
                    correct += 1
                total += 1
                
            if total > max_lim:
                break

    acc = correct / total if total > 0 else 0.0
    return acc


def train():
    model = LLaVACUBModel().to(DEVICE)
    model_raw = model  # Reference for non-DP methods

    if NUM_GPUS > 1:
        print(f"🚀 Utilizing {NUM_GPUS} Kaggle GPUs via DataParallel!")
        model = nn.DataParallel(model)

    """
        json_path,
        image_root,
        preprocess,
        tokenizer,
        image_attr_db=None,
        steering_prompt="Analyse bird image and output species and attributes:",
        max_seq_len=256,
        train_size=None
    """

    dataset = CUBDataset(json_path=JSON_PATH,image_root=IMAGE_ROOT,preprocess=model_raw.preprocess,tokenizer=model_raw.tokenizer,train_size=TRAIN_SIZE)
    val_dataset = CUBDataset(json_path=JSON_PATH_TEST,image_root=IMAGE_ROOT,preprocess=model_raw.preprocess,tokenizer=model_raw.tokenizer,train_size=TRAIN_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2 * max(1, NUM_GPUS), shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    total_steps = len(loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps)
    scaler = GradScaler()
    tracker = MetricsTracker()

    print(f"Dataset Size: {len(dataset)} | Total Steps: {total_steps}\n" + "─" * 60)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for step, (pv, prompt_ids, prompt_mask, target_ids, target_mask, _) in enumerate(loader):
            pv, prompt_ids, prompt_mask = pv.to(DEVICE), prompt_ids.to(DEVICE), prompt_mask.to(DEVICE)
            target_ids, target_mask = target_ids.to(DEVICE), target_mask.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                loss = model(pv, prompt_ids, prompt_mask, target_ids, target_mask)
                if NUM_GPUS > 1:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            tracker.log_step(loss.item(), current_lr)
            running_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{step+1}/{len(loader)}] Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        avg_loss = running_loss / len(loader)
        val_acc = evaluate_accuracy(model, val_loader, DEVICE)
        
        # Log metrics & save state
        tracker.log_epoch(avg_loss, eval_acc=val_acc)
        tracker.save()
        tracker.plot_and_save()

        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Extract unwrapped raw model
            active_model = model.module if NUM_GPUS > 1 else model
            
            checkpoint = {
                'epoch': epoch,
                # Strips the nn.DataParallel wrapper out
                'model_state_dict': active_model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, CKPT_PATH)
            print(f"DataParallel checkpoint saved to {CKPT_PATH}")
            # torch.save(save_payload, CKPT_PATH)
            
            # Also save PEFT adapter folder for easy Hugging Face reloading
            # active_model.llm.save_pretrained("./output/lora_adapters")
            
            print(f"💾 Checkpoint saved (Loss: {avg_loss:.4f}) -> {CKPT_PATH}\n")

    return model_raw, dataset


if __name__ == "__main__":
    model,dataset = train()