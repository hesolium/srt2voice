import numpy as np
from openai import OpenAI
from argparse import Namespace
import voiceUtils as vu
from voiceUtils import exitMsg
import io
import re

from pydub import AudioSegment
from pydub.silence import split_on_silence

args: Namespace = None
engineName = 'openai'
client = None
serverVoice = None
splitMarker: np.ndarray = None
patternDir = "omni-patterns"

import os
from contextlib import contextmanager

@contextmanager
def lock_voice_file(wav_path: str):
    # Otwieramy istniejący plik .wav tylko do odczytu - nie modyfikujemy go
    with open(wav_path, "rb") as f:
        if os.name == "nt":
            import msvcrt
            # Blokada 1 bajtu na Windows (czeka aż poprzedni proces zwolni)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            # Blokada wyłączna na Linux/macOS (automatycznie zwalniana przy wyjściu z with)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def joinTexts(text: list):
    ltext = ''
    for t in text:
        if ltext == '':
            ltext = t
        else:
            char = ''
            if ltext[-1] not in ".,!?":
                char = '.'
            ltext += char
            ltext += char + args.splitword + t
    ltext = re.sub(r'([?!])', r' \1', ltext)
    return ltext

def generate_contextual_marker_mfcc(voiceName) -> np.ndarray:
    marker = args.splitword.strip()
    if not marker.endswith('.'):
        marker += '.'

    # --- ETAP 1: Surowy marker (wygenerowany samodzielnie) ---
    raw_marker_audio = vu.trimSilence(generateVoice(marker))
    raw_mfcc = vu.normalized_mfcc(vu.audiosegment_to_float32(raw_marker_audio))

    # --- ETAP 2: Generacja w naturalnym kontekście zdań ---
    args.splitword = ' \n' + marker + '\n '
    contextual_insert = '... \n' + marker + '\n ...'
    template = args.splitcontext.replace(marker, contextual_insert)
    if template == args.splitcontext:
        print("Marker", marker, "not found in", args.splitcontext)
        sys.exit(1)

    template_audio = generateVoice(template)

    y_template = vu.audiosegment_to_float32(template_audio)
    template_mfcc = vu.normalized_mfcc(y_template)

    # Szukamy markera w szablonie (rozpakowujemy 2 wartości: similarity oraz best_scales)
    similarity, best_scales = vu.fast_correlate_2d_valid(
        template_mfcc, raw_mfcc, scales=(0.85, 0.95, 1.0, 1.05)
    )

    # Ograniczamy szukanie do środkowych 70% nagrania (ignorujemy pierwsze 15% i ostatnie 15% szablonu)
    n_sim = len(similarity)
    margin = int(n_sim * 0.15)
    if n_sim > 2 * margin:
        search_slice = similarity[margin: n_sim - margin]
        best_frame_idx = margin + int(np.argmax(search_slice))
    else:
        best_frame_idx = int(np.argmax(similarity))

    ms_per_frame = vu.WIN_STEP * 1000
    # Bezpieczne pobranie liczby ramek (drugi wymiar macierzy MFCC)
    raw_frames = raw_mfcc.shape[1]
    raw_duration_ms = int(raw_frames * ms_per_frame)

    # Uwzględniamy faktyczną skalę dopasowania z Etapu 2
    matched_scale = float(best_scales[best_frame_idx])
    actual_duration_ms = int(round(raw_duration_ms * matched_scale))
    #---------------------------------------------------------------------------
    # Poprawiony środek z uwzględnieniem actual_duration_ms
    center_ms = int(best_frame_idx * ms_per_frame) + (actual_duration_ms // 2)
    half_win_ms = int(actual_duration_ms * 0.70)
    start_ms = max(0, center_ms - half_win_ms)
    end_ms = min(len(template_audio), center_ms + half_win_ms)

    window_audio = template_audio[start_ms:end_ms]

    # Zostawiamy po 40 ms ciszy/wydechu z każdej strony wzorca (keep_silence=40)
    speech_parts = split_on_silence(
        window_audio, min_silence_len=60, silence_thresh=window_audio.dBFS - 18, keep_silence=35
    )
    if speech_parts:
        contextual_marker_audio = max(speech_parts, key=len)
    else:
        contextual_marker_audio = vu.trimSilence(window_audio)
    if args.keepFragments is not None:
        fl = 'marker-lektor-' + voiceName
        fl = os.path.join(args.fragmentsDir, fl)
        contextual_marker_audio.export(fl + ".wav", format="wav")
    return vu.normalized_mfcc(vu.audiosegment_to_float32(contextual_marker_audio))

def getVoiceReference(pargs: Namespace):
    sdir = os.path.dirname(pargs.config)
    odir = os.path.join(sdir, patternDir)
    if not os.path.exists(odir):
        if not pargs.frozen is None:
            odir = os.path.join(pargs.frozen, patternDir)
            if not os.path.exists(odir):
                exitMsg(patternDir, "directory not found")
    langPattern = os.path.join(odir, pargs.language)
    voicePattern = None
    ref_text = None
    voice = None
    if pargs.voice is not None:
        voicePattern = langPattern + '-' + pargs.voice
        if not os.path.exists(voicePattern + ".wav"):
            exitMsg("No reference audio for voice", pargs.language + '-' + pargs.voice)
        voice = pargs.voice
        if os.path.exists(voicePattern + ".txt"):
            ref_text = voicePattern + ".txt"
    else:
        if not os.path.exists(langPattern + ".wav"):
            exitMsg("No reference audio for language", pargs.language)
        voice = pargs.language
        voicePattern = langPattern
        if os.path.exists(voicePattern + ".txt"):
            ref_text = voicePattern
    if ref_text is None:
        if not os.path.exists(langPattern + ".txt"):
            exitMsg("No reference text for language", pargs.language)
        ref_text = langPattern
    with open(ref_text + ".txt", 'r', encoding="utf-8") as f:
        ref_text = f.read()
    return voice, voicePattern + '.wav', ref_text

def voice2Server():
    # 1. Pobieramy listę zapisanych profili z serwera
    voice, refSound, refText = getVoiceReference(args)
    with lock_voice_file(refSound):
        global splitMarker, serverVoice
        odpowiedz = client.get("/voices", cast_to=dict)

        # Serwer zwraca listę słowników lub obiekt z kluczem "profiles"
        profile = odpowiedz if isinstance(odpowiedz, list) else odpowiedz.get("voices", [])
        # istniejace_id = [
        #     p["profile_id"]
        #     for p in profile
        #     if isinstance(p, dict) and p.get("type") == "clone" and "profile_id" in p
        # ]
        istniejace_id = [p.get("profile_id") for p in profile if isinstance(p, dict)]

        # 2. Jeśli profil już tam jest, nic nie robimy!
        if not voice in istniejace_id:
            with open(refSound, "rb") as audio_file:
                response = client._client.post(
                    "/voices/profiles",
                    # cast_to=dict,
                    data={
                        "profile_id": voice,
                        "ref_text": refText,
                        "overwrite": "true"
                    },
                    files={"ref_audio": (voice + ".wav", audio_file, "audio/wav")}
                    # extra_headers={"Content-Type": "multipart/form-data"}
                )
                # response.raise_for_status()
            print("New voice", voice, "send to server")
        serverVoice = voice
        if args.batchMergeMode and hasattr(args, "splitword") and args.splitword is not None:
            splitMarker = generate_contextual_marker_mfcc(voice)

def generateVoice(text):
    if serverVoice is None or (args.voice is not None and args.voice != serverVoice):
        voice2Server()
    tcount = 1
    ltext = text
    if isinstance(text, list):
        tcount = len(text)
        if args.batchMergeMode and splitMarker is not None:
            ltext = joinTexts(text)

    with client.audio.speech.with_streaming_response.create(
            model=args.model, voice=serverVoice, input=ltext, response_format="wav"
    ) as response:
        audio_bytes = io.BytesIO(response.read())
        segment = AudioSegment.from_file(audio_bytes, format="wav")
    if tcount > 1 and args.batchMergeMode:
        segment = vu.splitVoiceByMarker(segment, splitMarker, tcount - 1)
    return segment

def init(progArgs):
    global args, client
    args = progArgs
    if args.model is None:
        args.model = 'omnivoice'
    if args.host is None:
        args.host = 'http://127.0.0.1:8880/v1'
    if args.key is None:
        args.key = 'xx'
    client = OpenAI(base_url=args.host, api_key=args.key)
