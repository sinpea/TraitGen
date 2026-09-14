import os
import json
from torch.utils.data import Dataset
from PIL import Image

from config import cub_root,STEERING_PROMPT

def parse_raw_attribute(raw_attr):
    """
    Converts CUB string 'has_crown_color::black' into ('crown', 'color', 'black').
    """
    try:
        # Split on '::'
        prefix, value = raw_attr.split("::")
        # Remove 'has_' and split body part and feature type
        prefix = prefix.replace("has_", "")
        parts = prefix.split("_")
        
        part_name = parts[0]          # e.g., 'crown', 'bill'
        attr_type = "_".join(parts[1:]) # e.g., 'color', 'shape'
        
        return {
            "part": part_name,
            "type": attr_type,
            "value": value.replace("_", " ")
        }
    except Exception:
        return None
    
def load_image_specific_attributes(
    cub_root_dir, 
    min_certainty=3  # 1: not visible, 2: guessing, 3: probably, 4: definitely
):
    """
    Parses CUB attributes and returns a map of image_id -> list of verified visual traits.
    """
    attr_names_path = os.path.join(cub_root_dir, "attributes", "attributes.txt")
    attr_labels_path = os.path.join(cub_root_dir, "attributes", "image_attribute_labels.txt")

    # 1. Map attribute_id to human-readable names
    # e.g., 101 -> "has_bill_shape::cone"
    attr_map = {}
    with open(attr_names_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                attr_id = int(parts[0])
                raw_name = parts[1]
                attr_map[attr_id] = raw_name

    # 2. Parse per-image attribute annotations
    # File format: image_id attribute_id is_present certainty_id time
    image_attributes = {}
    
    with open(attr_labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
                
            img_id = int(parts[0])
            attr_id = int(parts[1])
            is_present = int(parts[2])
            certainty = int(parts[3])

            # Filter: must be present AND meet the certainty threshold
            if is_present == 1 and certainty >= min_certainty:
                if img_id not in image_attributes:
                    image_attributes[img_id] = []

                raw_attr = attr_map.get(attr_id)
                if raw_attr:
                    # Clean CUB naming: "has_crown_color::black" -> ("crown", "color", "black")
                    cleaned = parse_raw_attribute(raw_attr)
                    if cleaned:
                        image_attributes[img_id].append(cleaned)

    return image_attributes

def build_image_wise_caption(species_name, image_traits):
    """
    Builds a Chain-of-Thought caption conditioned strictly on 
    attributes verified present in the specific image.
    """
    species = species_name.split(".")[-1].replace("_", " ")
    fields = []

    for trait in image_traits:
        part = trait["part"]
        attr_type = trait["type"]
        val = trait["value"]

        if attr_type == "color":
            fields.append(f"Its {part} is {val} color")
        elif attr_type in ["shape", "pattern"]:
            fields.append(f"Bird has {val} {part}")
        else:
            fields.append(f"Its {part} is {val}")

    fields.append(f"Deduction: {species}")
    return "; ".join(fields) + ";"

class CUBDataset(Dataset):
    """
    Optimized CUB-200-2011 Dataset loader using image-specific 
    visual traits for grounded multimodal CoT training.
    """
    def __init__(
        self,
        json_path,
        image_root,
        preprocess,
        tokenizer,
        image_attr_db=None,
        steering_prompt=STEERING_PROMPT,
        max_seq_len=256,
        train_size=None
    ):
        self.image_root = image_root
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.image_attr_db = image_attr_db or load_image_specific_attributes(cub_root, min_certainty=3)

        # 1. Parse JSON annotation metadata
        with open(json_path, "r") as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        self.categories = {cat["id"]: cat for cat in data["categories"]}
        self.samples = []

        for ann in data["annotations"]:
            img_id = ann["image_id"]
            img_info = self.images.get(img_id)
            cat_info = self.categories.get(ann["category_id"])

            if img_info and cat_info:
                self.samples.append({
                    "image_id": img_id,
                    "filename": img_info["filename"],
                    "species": cat_info["name"]
                })

        if train_size is not None:
            self.samples = self.samples[:train_size]

        # 2. OPTIMIZATION: Pre-tokenize steering prompt ONCE in __init__
        prompt_enc = self.tokenizer(
            steering_prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        self.prompt_ids = prompt_enc.input_ids.squeeze(0)
        self.prompt_mask = prompt_enc.attention_mask.squeeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_id = s["image_id"]

        # 1. Load and transform image
        img_path = os.path.join(self.image_root, s["filename"])
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.preprocess(image)  # [3, 224, 224]

        # 2. Fetch image-specific attributes (fallback to empty list if missing)
        image_traits = self.image_attr_db.get(img_id, [])

        # 3. Construct image-grounded target caption
        caption = build_image_wise_caption(s["species"], image_traits)
        caption_with_eos = f"{caption} {self.tokenizer.eos_token}"

        # 4. Tokenize target caption
        target_enc = self.tokenizer(
            caption_with_eos,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
            add_special_tokens=False
        )

        target_ids = target_enc.input_ids.squeeze(0)
        target_mask = target_enc.attention_mask.squeeze(0)

        # Return pre-tokenized prompt tensors directly
        return pixel_values, self.prompt_ids, self.prompt_mask, target_ids, target_mask, caption