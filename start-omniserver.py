import os
import sys
import random
import numpy as np
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from omnivoice import OmniVoice

# 1. Wymuszenie trybu offline przed importem serwera
os.environ["HF_HUB_OFFLINE"] = "1"

# 2. Twoje ustawienia determinizmu i TF32
def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

set_deterministic(42)

# 3. Przechwycenie ładowania modelu (Monkey-patching)
_orig_from_pretrained = OmniVoice.from_pretrained

def custom_from_pretrained(*args, **kwargs):
    print("Ładowanie modelu z niestandardowymi parametrami (float16, offline, bez ASR)...")
    kwargs["device_map"] = "cuda:0"
    kwargs["dtype"] = torch.float16
    kwargs["load_asr"] = False
    kwargs["local_files_only"] = True

    model = _orig_from_pretrained(*args, **kwargs)
    model.eval()
    
    print("Kompilacja modelu (torch.compile)...")
    model = torch.compile(model, mode="reduce-overhead")
    print("Model gotowy do pracy.")
    return model

OmniVoice.from_pretrained = custom_from_pretrained

# 4. Uruchomienie oryginalnego CLI z pakietu omnivoice-server
if __name__ == "__main__":
    # Możesz przekazać standardowe argumenty serwera, np. --host, --port, --device
    if len(sys.argv) == 1:
        sys.argv.extend(["--host", "0.0.0.0", "--port", "8880", "--device", "cuda"])
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
    VOICES_DIR = os.path.join(BASE_DIR, "voices")
    os.makedirs(PROFILES_DIR, exist_ok=True)
    os.makedirs(VOICES_DIR, exist_ok=True)
    # Dodajemy parametry linii poleceń przed wywołaniem main() z omnivoice_server
    sys.argv.extend([
        "--profile-dir", PROFILES_DIR,
        "--voice-dir", VOICES_DIR,
    ])
    from omnivoice_server.cli import main
    sys.exit(main())
