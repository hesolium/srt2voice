import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_nonsilent
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter
from operator import itemgetter
import re
import sys

SR = 16000
WIN_STEP = 0.010  # Krok ramki: 0.010 s (10 ms) -> odpowiednik HOP_LENGTH=160
WIN_LEN = 0.025   # Długość okna analizy: 25 ms (standard w rozpoznawaniu mowy)

def exitMsg(*lst, returnValue=1):
    if lst:
        print(*lst)
    sys.exit(returnValue)

def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
         ceplifter=22, appendEnergy=True):
    highfreq = highfreq or samplerate / 2

    # 1. Pre-emphasis
    signal = np.append(signal[0], signal[1:] - preemph * signal[:-1])

    # 2. Framing
    slen = len(signal)
    frame_len = int(round(winlen * samplerate))
    frame_step = int(round(winstep * samplerate))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(np.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)
    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((signal, zeros))

    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + \
              np.tile(np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    frames = padsignal[indices.astype(np.int32, copy=False)]

    # 3. Power Spectrum
    complex_spec = np.fft.rfft(frames, nfft)
    pow_spec = 1.0 / nfft * np.square(np.abs(complex_spec))
    energy = np.sum(pow_spec, 1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)

    # 4. Mel Filterbank
    lowmel = 2595 * np.log10(1 + lowfreq / 700.0)
    highmel = 2595 * np.log10(1 + highfreq / 700.0)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    hzpoints = 700 * (10 ** (melpoints / 2595.0) - 1)
    bin_points = np.floor((nfft + 1) * hzpoints / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin_points[j]), int(bin_points[j + 1])):
            fbank[j, i] = (i - bin_points[j]) / (bin_points[j + 1] - bin_points[j])
        for i in range(int(bin_points[j + 1]), int(bin_points[j + 2])):
            fbank[j, i] = (bin_points[j + 2] - i) / (bin_points[j + 2] - bin_points[j + 1])

    feat = np.dot(pow_spec, fbank.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    feat = np.log(feat)

    # 5. DCT-II (zastępuje scipy.fftpack.dct(..., norm='ortho'))
    N = feat.shape[1]
    n = np.arange(N)
    k = np.arange(numcep)[:, None]
    dct_mat = np.cos(np.pi * k * (2 * n + 1) / (2.0 * N)) * np.sqrt(2.0 / N)
    dct_mat[0] *= np.sqrt(0.5)
    feat = np.dot(feat, dct_mat.T)

    # 6. Liftering
    if ceplifter > 0:
        nframes, ncoeff = np.shape(feat)
        n = np.arange(ncoeff)
        lift = 1 + (ceplifter / 2.0) * np.sin(np.pi * n / ceplifter)
        feat *= lift

    if appendEnergy:
        feat[:, 0] = np.log(energy)

    return feat

def audiosegment_to_float32(seg: AudioSegment, target_sr: int = SR) -> np.ndarray:
    seg_norm = seg.set_frame_rate(target_sr).set_channels(1)
    samples = np.array(seg_norm.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(seg_norm.array_type).max
    return samples

def trimSilence(
        seg: AudioSegment,
        silence_thresh: int = -48,
        chunk_size: int = 10,
        pad_ms: int = 40
) -> AudioSegment:
    if len(seg) == 0:
        return seg

    start_trim = detect_leading_silence(seg, silence_threshold=silence_thresh, chunk_size=chunk_size)
    end_trim = detect_leading_silence(seg.reverse(), silence_threshold=silence_thresh, chunk_size=chunk_size)

    duration = len(seg)
    if start_trim + end_trim >= duration:
        return AudioSegment.empty()

    # Zostawiamy po pad_ms naturalnego oddechu z przodu i z tyłu, żeby nie ucinać ataku spółgłosek
    safe_start = max(0, start_trim - pad_ms)
    safe_end = min(duration, duration - end_trim + pad_ms)

    trimmed = seg[safe_start:safe_end]
    if len(trimmed) > 30:
        trimmed = trimmed.fade_in(15).fade_out(15)
    return trimmed

def normalized_mfcc(y: np.ndarray) -> np.ndarray:
    feat = mfcc(y, samplerate=SR, winlen=WIN_LEN, winstep=WIN_STEP, numcep=16, nfft=512).T[1:]
    return (feat - np.mean(feat, axis=1, keepdims=True)) / (np.std(feat, axis=1, keepdims=True) + 1e-8)

def simple_find_peaks(x: np.ndarray, height: float, distance: int):
    if len(x) < 3:
        return np.array([], dtype=int), np.array([])
    kernel = np.ones(5, dtype=np.float32) / 5.0
    x = np.convolve(x, kernel, mode="same")

    # ((x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:]))
    (candidates,) = np.where((x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] >= height))
    candidates = candidates + 1
    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([])
    order = candidates[np.argsort(x[candidates])[::-1]]
    selected = []
    for idx in order:
        if all(abs(idx - s) >= distance for s in selected):
            selected.append(idx)
    selected = np.sort(selected)
    return selected, x[selected]

def mfccMarker(marker_seg: AudioSegment) -> np.ndarray:
    clean_marker = trimSilence(marker_seg)
    y = audiosegment_to_float32(clean_marker, SR)
    return normalized_mfcc(y)

def fast_correlate_2d_valid(
        long_mfcc: np.ndarray,
        pat_mfcc: np.ndarray,
        scales: tuple[float, ...] = (0.80, 0.90, 1.0, 1.12, 1.25)
) -> tuple[np.ndarray, np.ndarray]:
    n_features, n_long = long_mfcc.shape
    _, base_n_pat = pat_mfcc.shape

    base_out_len = n_long - base_n_pat + 1
    if base_out_len <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    long_sq_cumsum = np.zeros((n_features, n_long + 1), dtype=np.float64)
    long_sq_cumsum[:, 1:] = np.cumsum(long_mfcc.astype(np.float64) ** 2, axis=1)

    fft_size = 1 << (n_long - 1).bit_length()
    fft_long = np.fft.rfft(long_mfcc, n=fft_size, axis=1)

    best_similarity = np.full(base_out_len, -1.0, dtype=np.float32)
    best_scales = np.ones(base_out_len, dtype=np.float32)

    for scale in scales:
        scaled_pat = _resize_mfcc_time(pat_mfcc, scale)
        _, n_pat = scaled_pat.shape
        out_len = n_long - n_pat + 1
        if out_len <= 0:
            continue
        win_sq_sum = np.sum(long_sq_cumsum[:, n_pat: n_pat + out_len] - long_sq_cumsum[:, :out_len], axis=0)
        win_norm = np.sqrt(np.maximum(win_sq_sum, 1e-8))
        pat_norm = np.linalg.norm(scaled_pat)
        if pat_norm < 1e-8:
            continue

        fft_pat = np.fft.rfft(scaled_pat[:, ::-1], n=fft_size, axis=1)
        conv = np.fft.irfft(fft_long * fft_pat, n=fft_size, axis=1)
        dot_prod = np.sum(conv[:, n_pat - 1: n_pat - 1 + out_len], axis=0)

        sim = (dot_prod / (win_norm * pat_norm)).astype(np.float32)

        offset = (n_pat - base_n_pat) // 2
        if offset >= 0:
            valid_len = min(base_out_len, len(sim) - offset)
            if valid_len > 0:
                slice_sim = sim[offset: offset + valid_len]
                target_sim = best_similarity[:valid_len]
                target_sc = best_scales[:valid_len]
                mask = slice_sim > target_sim
                target_sim[mask] = slice_sim[mask]
                target_sc[mask] = scale
        else:
            start_idx = -offset
            valid_len = min(base_out_len - start_idx, len(sim))
            if valid_len > 0:
                slice_sim = sim[:valid_len]
                target_sim = best_similarity[start_idx: start_idx + valid_len]
                target_sc = best_scales[start_idx: start_idx + valid_len]
                mask = slice_sim > target_sim
                target_sim[mask] = slice_sim[mask]
                target_sc[mask] = scale

    return best_similarity, best_scales

def _has_voice_energy(y_slice: np.ndarray, min_samples: int, min_rms: float = 0.005) -> bool:
    """Sprawdza na tablicy float32, czy fragment ma minimalną długość i energię mowy (ok. -46 dBFS)."""
    if y_slice.size < min_samples:
        return False
    rms = float(np.sqrt(np.mean(np.square(y_slice))))
    return rms >= min_rms

def _find_peaks_in_segment(
        segment: AudioSegment,
        marker_mfcc: np.ndarray,
        threshold: float,
        scales: tuple[float, ...] = (0.80, 0.90, 1.0, 1.12, 1.25)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    _, marker_frames = marker_mfcc.shape
    ms_per_frame = WIN_STEP * 1000
    marker_duration_ms = int(marker_frames * ms_per_frame)

    if len(segment) < int(marker_duration_ms * 0.75):
        return np.array([], dtype=int), np.array([], dtype=np.float32), np.array([],
                 dtype=np.float32), marker_duration_ms

    y = audiosegment_to_float32(segment, SR)
    mfcc_seg = normalized_mfcc(y)

    similarity, best_scales = fast_correlate_2d_valid(mfcc_seg, marker_mfcc, scales=scales)
    if len(similarity) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32), np.array([],
                dtype=np.float32), marker_duration_ms

    min_dist = int(marker_frames * 1.1)
    peaks, heights = simple_find_peaks(similarity, height=threshold, distance=min_dist)
    if len(peaks) == 0:
        return peaks, heights, np.array([], dtype=np.float32), marker_duration_ms

    peak_scales = best_scales[peaks]

    # --- FILTRACJA FAŁSZYWYCH I BRZEGOWYCH PIKÓW ---
    valid_indices = []
    # Minimalna długość sensownej wypowiedzi po obu stronach markera (np. 120 ms)
    min_speech_samples = int(0.120 * SR)
    # Margines, jaki _cut_segment_by_peaks zabiera wokół markera (marker_pad_ms=45 + bufor)
    pad_samples = int(0.060 * SR)

    for idx, (p, sc) in enumerate(zip(peaks, peak_scales)):
        start_sample = int(p * WIN_STEP * SR)
        actual_marker_samples = int(round((marker_duration_ms / 1000.0) * sc * SR))
        end_sample = start_sample + actual_marker_samples

        # 1. Odrzuć pik, jeśli sam domniemany marker jest ciszą (artefakt normalized_mfcc)
        marker_slice = y[max(0, start_sample):min(len(y), end_sample)]
        if not _has_voice_energy(marker_slice, min_samples=int(actual_marker_samples * 0.5), min_rms=0.008):
            continue

        # 2. Odrzuć pik, jeśli po lewej stronie (przed markerem i jego marginesem) nie ma mowy
        left_end = max(0, start_sample - pad_samples)
        if not _has_voice_energy(y[:left_end], min_samples=min_speech_samples, min_rms=0.005):
            continue

        # 3. Odrzuć pik, jeśli po prawej stronie (za markerem i jego marginesem) nie ma mowy
        right_start = min(len(y), end_sample + pad_samples)
        if not _has_voice_energy(y[right_start:], min_samples=min_speech_samples, min_rms=0.005):
            continue

        valid_indices.append(idx)

    if not valid_indices:
        return np.array([], dtype=int), np.array([], dtype=np.float32), np.array([],
                 dtype=np.float32), marker_duration_ms

    return peaks[valid_indices], heights[valid_indices], peak_scales[valid_indices], marker_duration_ms

# def _find_peaks_in_segment_OLD(
#         segment: AudioSegment,
#         marker_mfcc: np.ndarray,
#         threshold: float,
#         scales: tuple[float, ...] = (0.80, 0.90, 1.0, 1.12, 1.25)
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
#     _, marker_frames = marker_mfcc.shape
#     ms_per_frame = WIN_STEP * 1000
#     marker_duration_ms = int(marker_frames * ms_per_frame)
#
#     if len(segment) < int(marker_duration_ms * 0.75):
#         return np.array([], dtype=int), np.array([], dtype=np.float32), np.array([],
#                  dtype=np.float32), marker_duration_ms
#
#     y = audiosegment_to_float32(segment, SR)
#     mfcc_seg = normalized_mfcc(y)
#
#     similarity, best_scales = fast_correlate_2d_valid(mfcc_seg, marker_mfcc, scales=scales)
#     if len(similarity) == 0:
#         return np.array([], dtype=int), np.array([], dtype=np.float32), np.array([],
#                 dtype=np.float32), marker_duration_ms
#
#     min_dist = int(marker_frames * 1.1)
#     peaks, heights = simple_find_peaks(similarity, height=threshold, distance=min_dist)
#     peak_scales = best_scales[peaks] if len(peaks) > 0 else np.array([], dtype=np.float32)
#     return peaks, heights, peak_scales, marker_duration_ms
#
def _snap_to_silence(
        segment: AudioSegment,
        search_start_ms: int,
        search_end_ms: int,
        step_ms: int = 10
) -> int:
    """Znajduje środek najcichszego okienka (minimum RMS) w zadanym przedziale."""
    seg_len = len(segment)
    start = max(0, search_start_ms)
    end = min(seg_len, search_end_ms)

    if end - start <= step_ms:
        return max(0, min(seg_len, (start + end) // 2))

    best_pos = (start + end) // 2
    lowest_rms = float("inf")

    for pos in range(start, end - step_ms + 1, step_ms):
        rms = segment[pos:pos + step_ms].rms
        if rms < lowest_rms:
            lowest_rms = rms
            best_pos = pos + (step_ms // 2)

    return best_pos

def _cut_segment_by_peaks(
        segment: AudioSegment,
        peaks_with_scales: list[tuple[int, float]],
        base_marker_duration_ms: int,
        marker_pad_ms: int = 45
) -> list[AudioSegment]:
    ms_per_frame = WIN_STEP * 1000
    sub_chunks = []
    current_start_ms = 0
    seg_len = len(segment)

    for frame_idx, scale in sorted(peaks_with_scales, key=itemgetter(0)):
        center_ms = int(frame_idx * ms_per_frame) + (base_marker_duration_ms // 2)
        actual_duration_ms = int(round(base_marker_duration_ms * scale))

        raw_start = center_ms - (actual_duration_ms // 2) - marker_pad_ms
        raw_end = center_ms + (actual_duration_ms // 2) + marker_pad_ms

        # Szukamy pauzy przed markerem (cofamy się aż o 120 ms w lewo od raw_start,
        # aby na pewno złapać przerwę PRZED głoską "Ba")
        actual_start_ms = _snap_to_silence(
            segment,
            search_start_ms=max(current_start_ms, raw_start - 120),
            search_end_ms=min(seg_len, raw_start + 30)
        )

        # Szukamy pauzy po markerze (do 120 ms w prawo od raw_end)
        actual_end_ms = _snap_to_silence(
            segment,
            search_start_ms=max(actual_start_ms, raw_end - 30),
            search_end_ms=min(seg_len, raw_end + 120)
        )

        chunk = segment[current_start_ms:actual_start_ms]
        sub_chunks.append(chunk)
        current_start_ms = actual_end_ms

    sub_chunks.append(segment[current_start_ms:])
    return sub_chunks

# def _has_speech(seg: AudioSegment, min_duration_ms: int = 120, min_dbfs: float = -45.0) -> bool:
#     # Odrzuć za krótkie (np. resztki markera lub sam padding ciszy)
#     if len(seg) < min_duration_ms:
#         return False
#     # Odrzuć cyfrową ciszę i cichy szum tła (MFCC potrafi w nich znaleźć fałszywe piki!)
#     if seg.dBFS == float('-inf') or seg.dBFS < min_dbfs:
#         return False
#     return True
#
def splitVoiceByMarker(
        longVoice: AudioSegment,
        marker_mfcc: np.ndarray,
        expected_splits: int,
        high_threshold: float = 0.50,
        min_threshold: float = 0.30
) -> list[AudioSegment]:
    if expected_splits <= 0:
        return [trimSilence(longVoice)]

    segments = [longVoice]
    splits_done = 0
    thresholds = [high_threshold, 0.42, 0.36, min_threshold]

    for thresh in thresholds:
        if splits_done >= expected_splits:
            break

        candidates: list[tuple[float, int, int, float]] = []
        marker_duration_ms = 0

        for seg_idx, seg in enumerate(segments):
            peaks, heights, peak_scales, marker_duration_ms = _find_peaks_in_segment(
                seg, marker_mfcc, threshold=thresh
            )
            for p, h, sc in zip(peaks, heights, peak_scales):
                candidates.append((float(h), seg_idx, int(p), float(sc)))

        if not candidates:
            continue

        #-----------------------------------------------------------
        # Zamiast ślepego: chosen = candidates[:needed]
        needed = expected_splits - splits_done
        candidates.sort(key=itemgetter(0), reverse=True)

        chosen = []
        chosen_frames_by_seg: dict[int, list[int]] = {}
        # Minimalny odstęp między środkami dwóch markerów (długość markera + min. 150 ms na wypowiedź)
        min_frame_gap = int((marker_duration_ms + 150) / (WIN_STEP * 1000))

        for h, seg_idx, frame_idx, sc in candidates:
            if len(chosen) >= needed:
                break
            prev_frames = chosen_frames_by_seg.get(seg_idx, [])
            if any(abs(frame_idx - pf) < min_frame_gap for pf in prev_frames):
                continue
            chosen_frames_by_seg.setdefault(seg_idx, []).append(frame_idx)
            chosen.append((h, seg_idx, frame_idx, sc))
        #-------------------------------------------------------------

        cuts_by_seg: dict[int, list[tuple[int, float]]] = {}
        for _, seg_idx, frame_idx, sc in chosen:
            cuts_by_seg.setdefault(seg_idx, []).append((frame_idx, sc))

        new_segments = []
        for seg_idx, seg in enumerate(segments):
            if seg_idx in cuts_by_seg:
                cut_parts = _cut_segment_by_peaks(
                    seg, cuts_by_seg[seg_idx], marker_duration_ms
                )
                new_segments.extend(cut_parts)
                splits_done += len(cut_parts) - 1
            else:
                new_segments.append(seg)

        segments = new_segments
    #
    # if splits_done != expected_splits:
    #     print(f"WARNING: Expected {expected_splits} markers, {splits_done} detected.")
    #
    return [trimSilence(s) for s in segments]

def compressSilence(
        segment: AudioSegment,
        silence_ratio: float = 0.5,  # 0.5 = zostaw 50% długości pauz (1.0 = bez zmian)
        min_silence_len: int = 120,  # Ignoruj krótkie pauzy wewnątrz słów (ms)
        silence_step: int = 1,
        min_kept_ms: int = 40,  # Minimalna długość pauzy po skróceniu (ms)
        max_kept_ms: int = 450,  # Maksymalna długość pauzy po skróceniu (ms)
        trim_edges: bool = True,  # Czy całkowicie wyciąć ciszę na początku i końcu klipu
        silence_thresh_offset: int = -18  # Czułość wykrywania ciszy względem głośności nagrania (dBFS)
) -> AudioSegment:
    """Proporcjonalnie skraca przerwy między wyrazami/zdaniami w AudioSegment."""
    # trim_edges = True
    # silence_ratio = pargs.minGapAccelerate
    if silence_ratio >= 1.0 and not trim_edges:
        return segment
    thresh = segment.dBFS + silence_thresh_offset
    voiceRanges = detect_nonsilent(
        segment, min_silence_len=min_silence_len,
        silence_thresh=thresh, seek_step=silence_step)
    if not voiceRanges:
        return segment
    output = AudioSegment.empty()
    prev_end = 0
    for idx, (start, end) in enumerate(voiceRanges):
        silence_duration = start - prev_end
        if silence_duration > 0:
            if idx == 0:
                # Cisza na samym początku nagrania
                if not trim_edges:
                    output += segment[0:start]
            else:
                # Cisza między słowami/zdaniami
                if silence_ratio < 1.0:
                    new_len = int(silence_duration * silence_ratio)
                    new_len = max(min_kept_ms, min(new_len, max_kept_ms))
                else:
                    new_len = silence_duration
                if new_len > 0:
                    half = new_len // 2
                    output += (segment[prev_end: prev_end + half] + segment[start - (new_len - half): start])
        speech_part = segment[start:end]
        if len(output) > 10 and len(speech_part) > 10:
            output = output.append(speech_part, crossfade=5)
        else:
            output += speech_part
        prev_end = end
    # Cisza na samym końcu nagrania
    if not trim_edges and prev_end < len(segment):
        output += segment[prev_end:]
    return output

def speedVoice(segment: AudioSegment, speed: float = 1.0) -> AudioSegment:
    """Przyspiesza (lub zwalnia) AudioSegment algorytmem WSOLA bez zmiany wysokości głosu."""
    if abs(speed - 1.0) < 0.01:
        return segment
    samples = np.array(segment.get_array_of_samples())
    channels = segment.channels
    if channels > 1:
        samples = samples.reshape((-1, channels)).T
    else:
        samples = samples[np.newaxis, :]
    max_val = float(1 << (8 * segment.sample_width - 1))
    samples_float = samples.astype(np.float32) / max_val
    reader = ArrayReader(samples_float)
    writer = ArrayWriter(channels=channels)
    tsm = wsola(channels=channels, speed=speed)
    tsm.run(reader, writer)
    processed = np.clip(writer.data, -1.0, 1.0)
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    target_dtype = dtype_map.get(segment.sample_width, np.int16)
    int_samples = (processed * (max_val - 1)).astype(target_dtype)
    if channels > 1:
        int_samples = int_samples.T.flatten()
    else:
        int_samples = int_samples.flatten()
    return segment._spawn(int_samples.tobytes())

def _count_english_syllables(text: str) -> int:
    """Przybliżone zliczanie sylab dla języka angielskiego (uwzględnia nieme 'e' i dyftongi)."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    total_syllables = 0
    for word in words:
        if len(word) <= 3:
            total_syllables += 1
            continue

        # Usuwamy nieme końcówki -es, -ed (gdy nie poprzedzone t/d), -e (gdy nie -le)
        word = re.sub(r'(?:[^laeiouy]es|(?<![td])ed|[^laeiouy]e)$', '', word)
        word = re.sub(r'^y', '', word)

        # Grupy samogłosek liczymy jako 1 sylabę
        vowel_groups = re.findall(r'[aeiouy]{1,2}', word)
        total_syllables += max(1, len(vowel_groups))

    return total_syllables

def _estimate_text_weight(clean: str, lang: str = "pl") -> float:
    """
    Oblicza wagę czasową napisu zależnie od języka ('pl', 'en' lub inny).
    """
    lang_code = lang
    # 1. Liczenie sylab zależnie od języka
    if lang_code == "pl":
        # W j. polskim samogłoski (poza 'i' przed inną samogłoską) wyznaczają sylaby
        # Odejmujemy zmiękczenia typu 'ia', 'ie', 'io', 'iu', 'ią', 'ię'
        soft_i = len(re.findall(r'i[aąeęoóu]', clean, flags=re.IGNORECASE))
        vowels = len(re.findall(r'[aąeęioóuy]', clean, flags=re.IGNORECASE))
        syllables = max(1, vowels - soft_i)
        digit_weight = 2.8  # Polskie liczebniki są długie (np. "osiemdziesiąt" = 4 sylaby)

    elif lang_code == "en":
        syllables = _count_english_syllables(clean)
        digit_weight = 1.8  # Angielskie liczebniki są krótsze (np. "eighty" = 2 sylaby)

    else:
        # Uniwersalny fallback dla innych języków (np. DE, ES, FR)
        syllables = len(re.findall(r'[aeiouyàáâäãåèéêëìíîïòóôöõøùúûüąęó]', clean, flags=re.IGNORECASE))
        syllables = max(1, syllables)
        digit_weight = 2.2

    # 2. Liczba cyfr (każda cyfra generuje kilka sylab mowy)
    digits = len(re.findall(r'\d', clean))

    # 3. Pauzy interpunkcyjne (w jednostkach równoważnych sylabom)
    comma_pauses = (clean.count(',') + clean.count(';') + clean.count(':')) * 2.0
    sentence_pauses = (clean.count('.') + clean.count('!') + clean.count('?') + clean.count('…')) * 3.5

    weight = float(syllables) + (digits * digit_weight) + comma_pauses + sentence_pauses

    # Zabezpieczenie dla bardzo krótkich wtrąceń bez samogłosek (np. "Hmm", "Psst", "w")
    return max(weight, len(clean) * 0.35, 1.0)

def findUnsplitSegments(
    segments: list[AudioSegment], subtitles: list[str], lang: str = "pl"
) -> list[dict]:
    """
    Dopasowuje M segmentów audio do N napisów (gdzie M <= N) metodą programowania dynamicznego.
    Zwraca listę słowników tylko dla segmentów, które zawierają >= 2 sklejone napisy.
    """
    M = len(segments)
    N = len(subtitles)

    if M == N:
        return []
    if M == 0 or M > N:
        raise ValueError(f"Nieprawidłowa liczba segmentów ({M}) względem napisów ({N}).")

    # Wyznaczamy tempo mowy (ms na jednostkę wagi) dla całego batcha
    seg_durations = [max(len(s), 1) for s in segments]
    sub_weights = [_estimate_text_weight(t, lang=lang) for t in subtitles]

    total_audio_ms = sum(seg_durations)
    total_weight = sum(sub_weights)
    ms_per_unit = total_audio_ms / total_weight

    # dp[i][j] = minimalny koszt dopasowania pierwszych 'i' segmentów do pierwszych 'j' napisów
    INF = float('inf')
    dp = [[INF] * (N + 1) for _ in range(M + 1)]
    parent = [[None] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for i in range(1, M + 1):
        actual_ms = seg_durations[i - 1]
        # Pozostawiamy miejsce dla pozostałych segmentów (każdy musi mieć min. 1 napis)
        min_j = i
        max_j = N - (M - i)
        for j in range(min_j, max_j + 1):
            # Sprawdzamy ile napisów (od k do j-1) mogło trafić do segmentu i-1
            for k in range(i - 1, j):
                if dp[i - 1][k] == INF:
                    continue
                expected_ms = sum(sub_weights[k:j]) * ms_per_unit
                # Błąd względny (kwadratowy, aby silnie karać duże rozbieżności tempa)
                rel_error = (actual_ms - expected_ms) / max(expected_ms, 150.0)
                cost = dp[i - 1][k] + (rel_error ** 2)

                if cost < dp[i][j]:
                    dp[i][j] = cost
                    parent[i][j] = k

    # Odtwarzanie ścieżki od końca
    merged_results = []
    curr_i, curr_j = M, N
    while curr_i > 0:
        prev_j = parent[curr_i][curr_j]
        seg_idx = curr_i - 1
        assigned_indices = list(range(prev_j, curr_j))
        if len(assigned_indices) > 1:
            merged_results.append({
                "segment_index": seg_idx,
                "subtitle_indices": assigned_indices,
                "subtitles": [subtitles[idx] for idx in assigned_indices],
                "actual_duration_ms": seg_durations[seg_idx]
            })
        curr_i -= 1
        curr_j = prev_j
    merged_results.reverse()
    return merged_results

def _resize_mfcc_time(mfcc: np.ndarray, scale: float) -> np.ndarray:
    """Szybkie skalowanie macierzy MFCC (features, frames) w osi czasu w czystym NumPy."""
    n_features, n_frames = mfcc.shape
    new_frames = max(1, int(round(n_frames * scale)))
    if new_frames == n_frames:
        return mfcc

    x_old = np.linspace(0.0, 1.0, n_frames)
    x_new = np.linspace(0.0, 1.0, new_frames)

    resized = np.empty((n_features, new_frames), dtype=mfcc.dtype)
    for i in range(n_features):
        resized[i] = np.interp(x_new, x_old, mfcc[i])
    return resized


