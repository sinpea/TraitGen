import torch

JSON_PATH = "/kaggle/input/datasets/ralphraout/cub-preprocessed-2/cub_train_split1.json"
JSON_PATH_TEST = "/kaggle/input/datasets/ralphraout/cub-preprocessed-2/cub_test_split1.json"
IMAGE_ROOT = "/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011/images"

CKPT_PATH = "llava_bioclip_cub.pt"
METRICS_PATH = "metrics.json"
PLOT_PATH = "training_curves.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count()

# Optimized for Kaggle Dual T4s (16GB VRAM each)
BATCH_SIZE = 4 * max(1, NUM_GPUS)  # Effective batch size = 16 on dual T4s
EPOCHS = 5
LR = 2e-4
WARMUP_RATIO = 0.05
MAX_SEQ_LEN = 256
TRAIN_SIZE = None

CLIP_ID = "hf-hub:imageomics/bioclip-2"
LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Modern, lightweight decoder

STEERING_PROMPT = "Analyse bird image and output species and attributes:"
cub_root = "/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011"