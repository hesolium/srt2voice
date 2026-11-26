import os
from argparse import Namespace
from openai_gener import getVoiceReference, generate_contextual_marker_mfcc, joinTexts

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import logging

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

logging.getLogger("omnivoice").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import torch
import torchaudio
import random

def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Wyłączenie optymalizacji wybierającej różne algorytmy w zależności od rozmiaru wejścia
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Wymuszenie deterministycznych jąder CUDA w całym PyTorch
    torch.use_deterministic_algorithms(True, warn_only=True)

# set_deterministic(42)
from pydub import AudioSegment
from omnivoice import OmniVoice
import gc
import sys
import voiceUtils as vu
from voiceUtils import exitMsg

args: Namespace = None

#Global vars accessible by external functions
engineName = 'omni'

_REF_AUDIO = None
_REF_TEXT = None
voiceName = None
voice_prompt = None
_MODEL = None
sample_rate = 0

splitMarker: np.ndarray = None

def stateGPU(label=""):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"{label} VRAM allocated by PyTorch: {allocated:.2f} MB | Reserved (Cache): {reserved:.2f} MB")
    free_memory, total_memory = torch.cuda.mem_get_info()
    print(f"{label} GPU state. Free: {free_memory / (1024 ** 2):.2f} MB / Total: {total_memory / (1024 ** 2):.2f} MB")

def configure():
    global _REF_AUDIO, _REF_TEXT, voice_prompt, voiceName, splitMarker
    voiceName, _REF_AUDIO, _REF_TEXT = getVoiceReference(args)
    voice_prompt = _MODEL.create_voice_clone_prompt(ref_audio=_REF_AUDIO, ref_text=_REF_TEXT)
    # voiceName = args.voice
    if args.batchMergeMode and hasattr(args, "splitword") and args.splitword is not None:
        splitMarker = generate_contextual_marker_mfcc(voiceName)

def unload_model():
    global _MODEL
    if _MODEL is not None:
        del _MODEL
        _MODEL = None
        gc.collect()
        torch.cuda.empty_cache()

def get_model() -> OmniVoice:
    global _MODEL, sample_rate
    if _MODEL is None:
        print("Load model to GPU ...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        _MODEL = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype=torch.float16,
            load_asr=False,
            local_files_only=True
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _MODEL.eval()
        print("Compile model ...")
        _MODEL = torch.compile(_MODEL, mode="reduce-overhead")
        print("Model ready to run.")
        sample_rate = getattr(_MODEL, 'sample_rate', 24000)
        stateGPU()
    return _MODEL

@torch.inference_mode()  # Kluczowe: blokuje alokację pamięci na gradienty!
def generateVoice(text) -> AudioSegment | None:
    tcount = 1
    ltext = text
    genOption = {}
    if isinstance(text, list):
        tcount = len(text)
        if args.batchMergeMode and splitMarker is not None:
            ltext = joinTexts(text)
            genOption = {"seed" : 2345, "audio_chunk_threshold" : 60.0, "audio_chunk_duration" : 60.0}
    try:
        model = get_model()  # Pobiera już załadowany model z VRAM
        if args.voice is not None and args.voice != voiceName:
            configure(args.voice)
        audio_list = model.generate(
            text=ltext, voice_clone_prompt=voice_prompt,
            language=args.language, position_temperature=5.0,
            class_temperature = 0.0,        # Zachowuje powtarzalność brzmienia markera
            **genOption
        # num_step=16
            # speed=1.15         # np. 15% szybciej
            # duration=5.1     # lub dokładny czas trwania z napisów SRT (nadpisuje speed)
        )
        # if tcount > 1:
        #     stateGPU("After generating " + str(tcount) + " combined texts.")
        if not isinstance(audio_list, (list, tuple)):
            audio_list = [audio_list]
        result = []
        for audio in audio_list:
            # audio = audio_list[0] if isinstance(audio_list, (list, tuple)) else audio_list
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = np.asarray(audio, dtype=np.float32)
            audio_np = audio_np.squeeze()
            audio_int16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
            segment = AudioSegment(
                data=audio_int16.tobytes(), sample_width=2,
                frame_rate=sample_rate, channels=1)
            result.append(segment)
    except Exception as e:
        exitMsg("Omnivoice exception\n", e)
    if tcount > 1 and args.batchMergeMode:
        rseg = vu.splitVoiceByMarker(segment, splitMarker, tcount - 1)
        # if len(rseg) != tcount:
        #     if pargs.fragmentsDir is not None:
        #         fl = os.path.join(pargs.fragmentsDir, "split-voice-failed.wav")
        #         segment.export(fl, format="wav")
        segment = rseg
    elif isinstance(text, list):
        segment = result
    del audio_list
    return segment

    if args.keepFragments is not None:
        fl = 'omni-lektor-' + args.language if voiceName is None else voiceName
        fl = os.path.join(args.fragmentsDir, fl)
        contextual_marker_audio.export(fl + ".wav", format="wav")

    return vu.normalized_mfcc(vu.audiosegment_to_float32(contextual_marker_audio))

# Init global variables
def init(progArgs):
    global args
    args = progArgs
    if args.batchMergeMode:
        set_deterministic()
    get_model()
    configure()

