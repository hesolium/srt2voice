#!/usr/bin/env python3
"""
Copyright (C) 2025 Henryk Szal (hesolium@gmail.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
import traceback
import glob
import numpy as np
import iso639

# hidden imports for other modules
import shlex
from argparse import Namespace
import edge_tts
import time
import asyncio
import atexit
import io
import voiceUtils as vu
from voiceUtils import exitMsg
from openai import OpenAI

if sys.version_info.major >= 3 and sys.version_info.minor >= 13:
    exitMsg('The program requires Python version 3.12 or lower.')

import os
verboseFlag = 0

if '--verbose' in sys.argv:
    index = sys.argv.index('--verbose')
    if index + 1 < len(sys.argv):
        opt = sys.argv[index + 1]
        if not opt.isdigit():
            verboseFlag = 1
        else:
            verboseFlag = int(opt)

def verbose(*fArgs):
    if verboseFlag > 0:
        print(*fArgs)
        if verboseFlag > 1:
            input("Press Enter to continue ...")

frozen = getattr(sys, '_MEIPASS', None)
if frozen is not None:
    if frozen not in os.environ["PATH"]:
        os.environ["PATH"] = frozen + os.pathsep + os.environ["PATH"]
    v = "LD_LIBRARY_PATH"
    old_var = f"{v}_ORIG"
    if old_var in os.environ:
        os.environ[v] = os.environ[old_var] + os.pathsep + os.environ[v]
    # print(os.environ["PATH"])
    fdir = os.getcwd()
    if fdir not in os.environ["PATH"]:
        os.environ["PATH"] = fdir + os.pathsep + os.environ["PATH"]
    if fdir not in sys.path:
        sys.path.append(fdir)

import argparse
import shutil
import subprocess
import mimetypes
from collections import namedtuple
from tempfile import TemporaryDirectory
from typing import Optional
from alive_progress import alive_bar
from pathlib import Path
import importlib.util
from pysrt import SubRipFile, SubRipItem
from pydub import AudioSegment as AdSeg, AudioSegment
from pydub.silence import split_on_silence, detect_leading_silence
import pysrt
from gtts import gTTS
from configparser import ConfigParser
import runpy

__version__ = "0.6.0"
MovieTrack = namedtuple('MovieTrack', 'idx, codec_type, codec_name, language, duration')
tmp_merge: str = "-tmpAudioMerge.wav"
tmp_2audios: str = "-tmp_merged2audios."
dir_handle:str = ""
nativeSpeech: Optional[AudioSegment] = None
lector: Optional[AudioSegment] = None
defFrameRate = 24000
args: Namespace = None
formatIdx = '03d'
subs: Optional[SubRipFile] = None
subsCount = 0
barStyle = {
    "bar":'classic',
    "spinner": None
}
usage = \
"Version " + __version__ + " Copyright (c) 2025\n"\
"Generate a voice-over audio based on a specified subtitle file\n\n"\
"Positional argument (.srt file name) is not obligatory\n"\
"Subtitles and/or original audio can be extracted directly from video file (-n option)\n"\

if frozen:
    usage += "\nTo process multiple files simultaneously, you can use the included 'run-batch.py' script.\n"\
        "Example usage: srt2voice -e run-batch.py <run control file>\n"\
        "Format of <run control file> will be show after run script with no arguments specified\n"

def isEmpty(s: str) -> bool:
    # if s and attr:
    #     s = getattr(s, attr, None)
    if s is None or not s:
        return True
    if type(s) is str:
        return s.isspace()
    elif type(s) in (int, float):
        return s == 0
    return len(s) == 0

def quiet(*fArgs, **kwargs):
    if verboseFlag > 0 or not args.quiet:
        print(*fArgs, **kwargs)

def keepFragment(sFrag: AdSeg, idx, merged):
    if args.fragments is not None:
        idx1 = idx + 1
        if len(args.fragments) == 0 or idx1 in args.fragments:
            fn = getFragmentPrefix(idx, merged)
            sFrag.export(fn + ".wav", format="wav")

def generate_gtts(text):
    if isinstance(text, list):
        text = text[0]
    maxTry = 3
    while maxTry > 0:
        try:
            sFrag = gTTS(text=text, lang=args.language, slow=False)
            mp3_fp = io.BytesIO()
            sFrag.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            segment = AdSeg.from_mp3(mp3_fp)
            return segment
        except OSError:
            if not args.quiet:
                print("Network error! Sleep for 5 seconds and try again...")
            time.sleep(5)
            maxTry -= 1
            if maxTry == 0:
                print("Generating voice for '" + text + "' failed")
                sys.exit(1)


def highToneRatio(audio_segment, cutoff_freq=1000):
    """
    Calculates the ratio of high-frequency power (> cutoff_freq)
    to total power using ONLY NumPy.
    """
    # 1. Convert pydub audio data to a NumPy array (float32 for precision)
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    fs = audio_segment.frame_rate  # Sampling frequency (e.g., 24000 or 44100)

    # 2. Perform 1D Real Fast Fourier Transform
    # np.fft.rfft is optimized for real-valued signals (it drops redundant negative frequencies)
    fft_vals = np.fft.rfft(samples)

    # 3. Calculate corresponding frequencies for the rfft output
    freqs = np.fft.rfftfreq(len(samples), 1 / fs)

    # 4. Calculate Power Spectrum (magnitude squared)
    powers = np.abs(fft_vals) ** 2

    # 5. Sum the power arrays to find the ratio
    total_power = np.sum(powers) + 1e-10  # Prevent division by zero
    high_power = np.sum(powers[freqs >= cutoff_freq])
    return float(high_power / total_power)

def preprocessGenVoice(sFrag):
    if sFrag is not None:
        sFrag = vu.trimSilence(sFrag)
        if args.gapAccelerate > 0.0:
            sFrag = vu.compressSilence(sFrag, args.gapAccelerate)
            if args.accelerate > 1.0:
                sFrag = vu.speedVoice(sFrag, args.accelerate)
    return sFrag

def createVoice(txt: str):
    sFrag : Optional[AudioSegment] = None
    if args.voiceGenerator is None:
        sFrag = generate_gtts(txt)
    else:
        sFrag = args.voiceGenerator.generateVoice(txt)
    return sFrag

def fragmentPreloaded(idx: int, merged: int):
    if args.usePreloaded:
        fn = getFragmentPrefix(idx, merged) + ".wav"
        if os.path.exists(fn):
            return AdSeg.from_wav(fn)
    return None

# Generate speech from text.
def genFragment(txt: str, idx: int, merged: int):
    sFrag = fragmentPreloaded(idx, merged)
    if sFrag is None:
        # file not preloaded. Get speech from TTS engine
        if args.voiceCache is not None:
            sFrag = args.voiceCache.pop(idx, None)
        if sFrag is None:
            sFrag = createVoice(txt)
        keepFragment(sFrag, idx, merged)
    if not args.preloadOnly:
        sFrag = preprocessGenVoice(sFrag)
    return sFrag

def getFragmentPrefix(idx: int, merged: int):
    idx1 = idx + 1
    fn = format(idx1, formatIdx)
    if merged > 0:
        fn += "-" + str(idx1 + merged)
    if args.subtitleBasename is not None:
        fn = args.subtitleBasename + "-" + fn
    fl = os.path.join(args.fragmentsDir, fn)
    return fl

def fit2subtitle(
        segment: AudioSegment,
        target_duration_ms: int,
        min_silence_ratio: float = 0.4,  # Graniczna wartość kompresji ciszy dla danego głosu (np. 40%)
        max_speed: float = 1.35,  # Graniczna wartość przyspieszenia mowy dla danego głosu (np. 135%)
        trim_edges: bool = True
) -> AudioSegment:
    """
    Automatycznie dopasowuje nagranie do czasu trwania napisu (target_duration_ms),
    respektując graniczne parametry ustalone dla konkretnego głosu.
    """
    # Krok 1: Usuwamy puste brzegi (ciszę przed i po wypowiedzi)
    current = vu.compressSilence(
        segment, silence_ratio=1.0, trim_edges=trim_edges,
        min_silence_len=args.silenceLength, silence_step=args.silenceStep)
    if len(current) <= target_duration_ms:
        return current
    # Krok 2: Sprawdzamy, ile daje maksymalna dozwolona kompresja ciszy dla tego głosu
    compressed = vu.compressSilence(current, silence_ratio=min_silence_ratio, trim_edges=trim_edges)
    if len(compressed) <= target_duration_ms:
        # Skoro pełna kompresja wystarczyła z zapasem, możemy użyć łagodniejszej kompresji ciszy
        total_silence = len(current) - len(vu.compressSilence(current, silence_ratio=0.0, min_kept_ms=0))
        if total_silence > 0:
            needed_reduction = len(current) - target_duration_ms
            optimal_ratio = max(min_silence_ratio, 1.0 - (needed_reduction / total_silence))
            return vu.compressSilence(current, silence_ratio=optimal_ratio, trim_edges=trim_edges)
        return compressed
    # Krok 3: Cisza skompresowana na maksa, resztę nadrabiamy przyspieszeniem mowy (do granicy max_speed)
    required_speed = len(compressed) / float(target_duration_ms)
    final_speed = min(required_speed, args.maxAccelerate)
    return vu.speedVoice(compressed, speed=final_speed)

def generateFragment(idx: int):
    global args, subs
    sb = getSubtitle(idx)
    txt = sb.text
    verbose("Subtitle", idx + 1, ':', txt)
    verbose("Time range: [", sb.start, ",", sb.end, "]  Duration:", sb.duration.ordinal, "ms")
    skipSub = 0
    sFrag = genFragment(txt, idx, skipSub)
    if args.preloadOnly:
        return None, 0, 0
    sb2 = getSubtitle(idx + 1)
    orgLen = len(sFrag)
    sgap = (sb2.start - sb.start).ordinal if sb2 else orgLen
    verbose("Speech segment len:", orgLen, "ms")
    if 0 < sgap < orgLen:
        # try to concatenate with next subtitle
        if sb2:
            txt2 = txt + ' ' + sb2.text
            skipSub = 1
            sFrag = genFragment(txt2, idx, skipSub)
            orgLen = len(sFrag)
            sb3 = getSubtitle(idx + 2)
            sgap = (sb3.start - sb.start).ordinal if sb3 else orgLen
            verbose("Gap too small. Concatenated sub:", txt2)
            verbose("Concatenated gap len:", sgap, "ms")
        if 0 < sgap < orgLen and (args.maxAccelerate > 1.0 or 0 < args.minGapAccelerate < 1.0):
            # concatenation failed. Speed up speech
            verbose("Concatenated speech too long (", orgLen, "ms). Speed up.")
            sFrag = fit2subtitle(sFrag, sgap, args.minGapAccelerate, args.maxAccelerate, trim_edges=True)
        verbose("New speech segment len:", len(sFrag), "ms")
    return sFrag, sgap, skipSub

def setFromConfig(langInfo):
    sat = [option for option in langInfo]
    for key in sat:
        latr = langInfo[key]
        if key == 'gapaccelerate':
            key = 'gapAccelerate'
        atr = getattr(args, key, None)
        if atr is None and latr is not None:
            setattr(args, key, latr)

def removeSilence(frag):
    if args.silence is None:
        return None
    chunks = split_on_silence(
        frag, min_silence_len=args.silenceLength, silence_thresh=args.silenceThreshold,
        keep_silence=False, seek_step=args.silenceStep)
    if len(chunks) == 0 or len(chunks[0]) == len(frag):
        return None
    lFragment = AdSeg.empty()
    for c in chunks:
        lFragment += c
    return lFragment

def decodeAccelerate(val:str):
    ma = ra = 0
    if not isEmpty(val):
        t = val.split(':')
        ma = float(t[0])
        if len(t) > 1 and not isEmpty(t[1]):
            ra = float(t[1])
            args.stretch = True
    return ma, ra

def printNativeTracks():
    print("Native media type:", args.nativeMediaType)
    if len(args.nativeStreams) > 1:
        print("Streams in media file:")
        for t in range(len(args.nativeStreams)):
            s: MovieTrack = args.nativeStreams[t]
            lg = str(s.language.part1) if s.language is not None else 'None'
            print('Track:', s.idx, 'type:', s.codec_type, 'lang:', lg)

def badStreamsInfo(msg):
    if msg:
        print(msg)
    if len(args.nativeStreams) > 1:
        printNativeTracks()
        print("You can use option -t to specify tracks for processing")
    sys.exit(1)

def importExternalModule(mName):
    p = Path(mName)
    if p.suffix == '.py':
        mName = p.stem
    mod = importlib.import_module(mName)
    return mod

def showOptions():
    print("Fundamental options for voice generation:")
    if args.shift != 0:
        print("\tShift audio:", str(args.shift), "ms")
    if args.volume != 0.0:
        if args.relative:
            print("\tAudio gain relative to original audio volume:", str(args.volume), 'dB')
        else:
            print("\tAudio gain:", str(args.volume), 'dB')
    print("\tMax speed up voice by", args.maxAccelerate, "ratio")
    print("\tMax narrow silence between words.", args.minGapAccelerate, "ratio")
    if args.silenceLength != 0:
        print("\tOption for remove long silence between words:")
        print("\t\tMinimal length:", str(args.silenceLength), "ms")
        print("\t\tThreshold:", str(args.silenceThreshold), "dB")
        print("\t\tStep:", str(args.silenceStep))

def processArguments():
    global subsCount, subs, dir_handle, args, tmp_merge, tmp_2audios, nativeSpeech, verbose
    args.verboseFile = None
    args.lectorTimbre = None
    args.frozen = frozen
    if args.verbose is not None:
        if not str(args.verbose).isdigit():
            args.verboseFile = open(args.verbose, "w", buffering=1)
            verboseFlag = 1
        else:
            verboseFlag = int(args.verbose)
    if args.config is None:
        cName = args.config = "srt2voice.ini"
        if not os.path.exists(args.config) and frozen is not None:
            args.config = os.path.join(frozen, args.config)
            cName = "internal srt2voice.ini"
        if not os.path.exists(args.config):
            args.config = None
    else:
        cName = args.config
    if args.config is None or not os.path.exists(args.config):
        if args.config is None:
            exitMsg("No INI file was specified and no standard (srt2voice.ini) was found.")
        exitMsg("Specified INI file '" + args.config + "' was not found.")
    if isEmpty(args.voiceGenerator):
        args.voiceEngine = 'gtts'
    else:
        args.voiceGenerator = importExternalModule(args.voiceGenerator)
        if hasattr(args.voiceGenerator, 'engineName'):
            args.voiceEngine = args.voiceGenerator.engineName
    config = ConfigParser()
    config.read(args.config)
    sections = 0
    if not isEmpty(args.voice): # optional section for voice
        sectionId = args.language + ':' + args.voiceEngine + ":" + args.voice
        if config.has_section(sectionId):
            sectInfo = config[sectionId]
            quiet("Process settings from " + cName + " ["+ sectionId + "] section.")
            setFromConfig(sectInfo)
            sections += 1
    sectionId = args.language + ':' + args.voiceEngine
    if config.has_section(sectionId):
        sectInfo = config[sectionId]
        quiet("Process settings from " + cName + " [" + sectionId + "] section.")
        setFromConfig(sectInfo)
        sections += 1
    sectionId = args.language
    if config.has_section(sectionId):
        sectInfo = config[sectionId]
        quiet("Process settings from " + cName + " [" + sectionId + "] section.")
        setFromConfig(sectInfo)
        sections += 1
    if sections == 0:
        exitMsg("No (sub)sections for " + args.language + ',' + args.voiceEngine + " in " + cName + " file")
    args.fragments = args.fragmentsDir = None
    if args.keepFragments is not None:
        args.fragments = {}
        idx = args.keepFragments.rfind(':')
        if idx >= 0:
            if idx > 0:
                args.fragmentsDir = args.keepFragments[:idx]
            flist = args.keepFragments[idx + 1:]
            if len(flist) > 0:
                flist = flist.split(',')
                args.fragments = {int(x) for x in flist}
        else:
            args.fragmentsDir = args.keepFragments
    if args.fragmentsDir is None:
        dir_handle = TemporaryDirectory()
        args.tempDir = dir_handle.name
        if args.fragments is not None:
            args.fragmentsDir = ""
    else:
        if not os.path.isdir(args.fragmentsDir):
            exitMsg("Specified directory '" + args.fragmentsDir + "' was not found.")
        args.tempDir = args.fragmentsDir
    if args.nativeMedia is not None:
        getNativeStreams()
    args.srtTrackIdx = args.audioTrackIdx = None
    if args.tracks is not None:
        tok = args.tracks.split(':') if ':' in args.tracks else [args.tracks]
        for t in tok:
            if t.startswith('s'):
                args.srtTrackIdx = int(t[1:])
                if args.srtTrackIdx >= len(args.nativeStreams) or args.nativeStreams[args.srtTrackIdx].codec_type != 's':
                    badStreamsInfo("Incorrect subtitle track '" + t + "' specified")
            elif t.startswith('a'):
                args.audioTrackIdx = int(t[1:])
                if args.audioTrackIdx >= len(args.nativeStreams) or args.nativeStreams[args.audioTrackIdx].codec_type != 'a':
                    badStreamsInfo("Incorrect audio track '" + t + "' specified")
    args.subtitleBasename = Path(args.srtFile).stem if args.srtFile else None
    args.outDir = None
    if args.outFile is not None and os.path.isdir(args.outFile):
        args.outDir = args.outFile
        args.outFile = None
    if args.outFile is None:
        sf = ''
        if args.srtFile is not None:
            args.outFile = Path(args.srtFile).stem
        elif args.nativeMedia is not None:
            args.outFile = Path(args.nativeMedia).stem
    else:
        sf = Path(args.outFile).suffix
    if args.outFile is None:
        exitMsg("Unable to determine output file name. Specify -o option")
    tmp_merge = os.path.join(args.tempDir, args.outFile + tmp_merge)
    tmp_2audios = os.path.join(args.tempDir, args.outFile + tmp_2audios)
    outExt = '.aac'
    if args.merge is not None:
        if 'v' in args.merge:
            outExt = Path(args.nativeMedia).suffix
        else:
            if args.nativeMediaType == 'v':
                if args.audioTrackIdx is None:
                    stream = getStream("a")
                else:
                    stream = args.nativeStreams[args.audioTrackIdx]
                outExt = '.' + stream.codec_name
            else:
                outExt = Path(args.nativeMedia).suffix
    if isEmpty(sf):
        if outExt == '.eac3':
            outExt = '.ac3'
        args.outFile += outExt
        args.outFileDeduced = True
    else:
        args.outFileDeduced = False
        if 'v' not in args.merge:
            # output file must be audio type
            mimetypes.init()
            ft = mimetypes.guess_type(args.outFile)
            if ft is not None:
                ft = ft[0].split('/')
                if ft[0] != 'audio':
                    exitMsg("Output file extension mismatch. Should be any type 'audio' supported by ffmpeg")
        elif sf != outExt:
            exitMsg("Video output file must have the same file extension as merged original media file")
    # Check if output file exists
    if args.outDir is not None:
        args.outFile = os.path.join(args.outDir, args.outFile)
    if os.path.exists(args.outFile) and os.path.getsize(args.outFile) > 0:
        exitMsg("Output file '" + args.outFile + "' already exists")
    if args.nativeMedia:
        if args.srtFile is None:
            # SRT from movie file?
            if args.srtTrackIdx is not None:
                track = args.nativeStreams[args.srtTrackIdx]
                if track.codec_type != 's':
                    badStreamsInfo('Specified subtitle track ' + str(args.srtTrackIdx) + " is not proper type (subtitle)")
            else:
                track = getStream("s", args.isoLang)
            if track is not None:
                # extract SRT from media file
                extractTrack(args.nativeMedia, track)
                args.subtitleBasename = Path(args.nativeMedia).stem
                if args.merge:
                    # Merge subtitle unnecessary
                    args.merge = args.merge.replace("s", "")
    if not isEmpty(args.formatter):
        glob = {'args': args}
        if args.formatter.endswith('.py'):
            args.formatter = args.formatter[:-3]
        args.clearSrt = args.srtFile
        runpy.run_module(args.formatter, init_globals=glob, run_name= "__main__")
        if args.clearSrt != args.srtFile:
            quiet("Cleaned SRT file saved in:", args.clearSrt)
            if args.srtFile is None:
                args.srtFile = args.clearSrt
        else:
            args.clearSrt = args.srtFile
    else:
        args.clearSrt = args.srtFile
    if args.merge and args.nativeMedia is None:
        exitMsg("Merge [-m] option specified but no native media file specified [-n].")
    # Preprocess options
    args.stretch = False
    args.maxAccelerate, args.accelerate = decodeAccelerate(str(args.accelerate))
    args.minGapAccelerate, args.gapAccelerate = decodeAccelerate(str(args.gapAccelerate))
    args.highFilterCompensation = 2.5  # dB
    args.highFilterFreq = 1000
    if args.highfilter is not None: # frequency in Hz/compensation in dB
        hf = args.highfilter.split('/')
        if(len(hf) > 1):
            args.highFilterCompensation = float(hf[1])
        args.highFilterFreq = int(hf[0])
    if args.volume is not None:
        args.relative = args.volume[0] == 'r'
        if args.relative:
            if args.nativeMedia is None:
                exitMsg("Sound amplification relative to audio track volume is specified but no native media file option [-n] in args")
            args.volume = args.volume[1:]
        args.volume = float(args.volume)
    else:
        args.volume = 0
    args.shift = 0 if args.shift is None else int(args.shift)
    if isEmpty(args.silence):
        args.silence = None
    if args.silence is None:
        args.silenceStep = 10
        args.silenceThreshold = -45
        args.silenceLength = 100
    else:
        sop = args.silence.split(':')
        args.silenceStep = int(sop[2]) if len(sop) > 2 else 1
        args.silenceThreshold = int(sop[1]) if len(sop) > 1 else -45
        args.silenceLength = int(sop[0]) if len(sop) > 0 else 100
    if args.clearSrt is not None:
        subs = pysrt.open(args.clearSrt)
        subsCount = len(subs)
    else:
        badStreamsInfo("No SRT source. SRT file name not given and no '" + args.language + "' subtitle in media file")
    if (args.relative or args.merge or args.keepFragments) and args.nativeMedia is not None and not args.preloadOnly:
        if args.audioTrackIdx is None:
            stream = getStream("a")
        else:
            stream = args.nativeStreams[args.audioTrackIdx]
        if stream.idx > 0:
            extractTrack(args.nativeMedia, stream)
            args.audioTrackIdx = stream.idx
        elif stream.idx == 0:
            if args.nativeMediaType != 'a':
                extractTrack(args.nativeMedia, stream)
            else:
                args.audioTrack = args.nativeMedia
        quiet("Load native audio file ... ", end='')
        nativeSpeech = AdSeg.from_file(args.audioTrack)
        quiet("loaded.")
    if not args.preloadOnly and not args.quiet:
        showOptions()
    args.batchMergeMode = False
    if args.batchSize is not None:
        tok = args.batchSize.split('/')
        try:
            args.batchSize = int(tok[0])
            if len(tok) > 1:
                args.batchMergeMode = tok[1][0] in "mM"
        except ValueError:
            exitMsg("Improper batchSize format. Must be <number>[/M]")
    else:
        args.batchSize = 0
    if args.voiceGenerator is not None and hasattr(args.voiceGenerator, "init"):
        args.voiceGenerator.init(args)

def cmdOutput(cmdArgs:list, progressTime=0, barTitle=None):
    verbose("Run command:", cmdArgs)
    p = subprocess.Popen(cmdArgs, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            bufsize=1, universal_newlines=True)
    stderrLines = []
    if progressTime > 0 and not args.quiet:
        mkv = (cmdArgs[0] == "mkvmerge")
        p_err = p.stderr
        p_out = p.stdout
        barInp = p_out if mkv else p_err
        with alive_bar(progressTime, title=barTitle, length=50, manual=True, **barStyle) as cbar:
            while True:
                line = barInp.readline()
                if not line:
                    break
                tm = None
                if mkv:
                    k = line.lower().find("progress")
                    if k >= 0:
                        proc = line[k + 9:]
                        proc = proc.strip().replace('%', '')
                        tm = int(proc) / 100
                elif 'size=' in line:
                    k = line.find(' time=')
                    if k > 0:
                        proc = line[k + 6:]
                        proc = proc.split()
                        if ':' in proc[0]:
                            tm = time2seconds(proc[0]) / progressTime
                if args.verboseFile is not None:
                    line = line.rstrip()
                    print(line, file=args.verboseFile, flush=True)
                if tm is None:
                    stderrLines.append(line)
                else:
                    cbar(tm)
            cbar(1.0)
        p.wait()
        if p.returncode != 0:
            p_err = "".join(stderrLines)
    else:
        p_out, p_err = p.communicate()
    if p.returncode != 0:
        print("\n**** Program execution error:", p.returncode)
        if not args.verbose:
            print(cmdArgs)
        msg = "\nProgram output:\n{0}".format(p_err)
        print(msg)
        traceback.print_stack()
        sys.exit(1)
    return p_out

def extractTrack(movieName, trackDes: MovieTrack):
    basename = Path(movieName).stem
    rest = " from media file. Track: " + str(trackDes.idx) + " Duration: " + str(int(trackDes.duration / 60)) + ":" + str(trackDes.duration % 60) + " min. "
    extraArgs = []
    if trackDes.codec_type == 's':
        args.srtFile = outFile = os.path.join(args.tempDir, basename + ".srt")
        tit ="Extract subtitle"
        extraArgs.append("-an")
    else:
        args.audioTrack = outFile = os.path.join(args.tempDir, basename + ".aac")
        tit ="Prepare reference audio"
        extraArgs += ["-ac", "1", "-sn"]
        # extraArgs.append("-sn")
    if args.usePreloaded and os.path.exists(outFile):
        quiet("File '" + outFile + "' already exists, skipping extraction")
        return "OK"
    car = ["ffmpeg", "-hide_banner", "-y", "-i", movieName, "-vn", "-map", "0:" + str(trackDes.idx)] + extraArgs + [outFile]
    tit += " to file '" + outFile + "'"
    tit += rest
    quiet(tit)
    out = cmdOutput(car, progressTime=trackDes.duration)
    if out is None:
        sys.exit(1)
    return out

def time2seconds(tm):
    t = tm.split(':')
    hours = int(t[0])
    minutes = int(t[1])
    sec = float(t[2])
    minutes += hours * 60
    sec += minutes * 60
    return int(sec)

def isNativeVideo():
    if args.nativeStreams:
        for t in args.nativeStreams:
            if t[1] == 'v':
                return True
    return False

def getDuration():
    # "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 wszech.mp4"
    car = [
        "ffprobe", "-hide_banner", "-show_entries", "format=duration", "-show_entries",
        "stream=index,codec_type,codec_name,duration:stream_tags=language,duration", "-of", "compact=p=0", args.nativeMedia
    ]

def getNativeStreams():
    car = [
        "ffprobe", "-hide_banner", "-show_entries", "format=duration", "-show_entries",
        "stream=index,codec_type,codec_name:stream_tags=language,duration", "-of", "compact=p=0", args.nativeMedia
    ]
    out = cmdOutput(car)
    if not isEmpty(out):
        fileDuration = 0
        args.nativeMediaType = 'a'
        args.nativeStreams = []
        lst = out.split()
        for t in lst:
            t = t.lower()
            t = t.replace('tag:', '')
            w = t.split('|')
            sd = dict()
            sd["language"] = sd["duration"] = None
            add = True
            for desc in w:
                key, val = desc.split("=")
                if key == 'codec_type':
                    val = val[:1]
                    if val == 'v':
                        args.nativeMediaType = val
                elif key == 'duration':
                    if ':' in val:
                        val = time2seconds(val)
                    else:
                        fileDuration = int(float(val))
                        add = False
                elif key == 'index':
                    val = int(val)
                    key = "idx"
                elif key == 'language':
                    val = iso639.Language.match(val)
                sd[key] = val
            if add:
                args.nativeStreams.append(MovieTrack(**sd))
        if fileDuration != 0:
            for it in range(len(args.nativeStreams)):
                if args.nativeStreams[it].duration is None:
                    # noinspection PyProtectedMember
                    args.nativeStreams[it] = args.nativeStreams[it]._replace(duration=fileDuration)
        if args.verbose:
            printNativeTracks()

def getStream(sType, lang=None) -> Optional[MovieTrack]:
    if args.nativeStreams:
        for t in args.nativeStreams:
            if t.codec_type == sType:
                if lang and t.language != lang:
                    continue
                return t
    return None

def getSubtitle(idx) -> Optional[SubRipItem]:
    if idx >= len(subs):
        return None
    s = subs[idx]
    s.text = ' '.join(s.text.split())
    return s

def setMetaLang(aList: list, trackIdx: int):
    aList += ["-metadata:s:" + str(trackIdx), "language=" + args.language]

def extractSource():
    system = os.name
    content = [
        'README', 'requirements.txt', 'srt2voice.py', 'edge-gener.py', 'make-release', 'srt2voice.ini',
        'cleanSubtitle.py', 'run-batch.py', 'mp4box', "omni-gener.py", "start-omniserver.py"
    ]
    if args.extract == "*":
        files = content
    else:
        files = args.extract.split(',')
    print("Extract files from package to current directory.")
    fidx = []
    for name in files:
        fidx.extend([index for index, item in enumerate(content) if name in item])
    fidx = list(dict.fromkeys(fidx))
    if len(fidx) == 0:
        print("No files to extract for specified pattern '" + args.extract + "'")
        print("Files to extract:", content)
    for idx in fidx:
        file = content[idx]
        print("Extract:", file)
        src = os.path.join(frozen, file)
        dst = file
        if system == "nt":
            if file == "mp4box":
                src += '.exe'
                dst += '.exe'
            if file == "make-release":
                src += '.bat'
                dst += '.bat'
        shutil.copy(src, dst)
    sys.exit(0)

def fitLectorVoice(lektor: AudioSegment, tlo_filmu: AudioSegment, offset_db: float = 7.0):
    # 1. Oblicz prawdziwą głośność mowy lektora (bez pauz > 80 ms)
    fragments = split_on_silence(
        lektor, min_silence_len=args.silenceLength,
        silence_thresh=args.silenceThreshold, seek_step=args.silenceStep,  keep_silence=False)
    if fragments:
        speech = sum(fragments)
        lectorVolume = speech.dBFS
    else:
        lectorVolume = lektor.dBFS
    if args.lectorTimbre is not None:
        movieTimbre = highToneRatio(tlo_filmu, cutoff_freq=args.highFilterFreq)
        diff = round(movieTimbre - args.lectorTimbre, 1)
        if diff > 0:
            diff *= args.highFilterCompensation
            offset_db += diff
            verbose("Lector timbre correction:", diff, "dB")
    # 2. Oblicz docelową głośność na podstawie tła filmu + zapas użytkownika
    movieVolume = tlo_filmu.dBFS
    targetVolume = movieVolume + offset_db
    # 3. Widełki bezpieczeństwa (żeby lektor nie szeptał w ciszy i nie krzyczał w akcji)
    targetVolume = max(-22.0, min(-10.0, targetVolume))
    # 4. Max limit (peak <= -1.0 dBFS)
    gain = targetVolume - lectorVolume
    if lektor.max_dBFS + gain > -1.0:
        gain = -1.0 - lektor.max_dBFS
    return lektor.apply_gain(gain)

def normalizeFragment(voice: AudioSegment, sentence, skip):
    # global
    if args.volume != 0.0:
        gain = args.volume
        lectorDB = round(voice.dBFS, 1)
        if nativeSpeech and args.relative:
            fEnd = sentence.end.ordinal
            if skip > 0:
                fEnd = subs[skip].end.ordinal
            if fEnd - sentence.start.ordinal > len(voice):
                fEnd = sentence.start.ordinal + len(voice)
            nFragment:AudioSegment = nativeSpeech[sentence.start.ordinal:fEnd]
            fragDB = round(nFragment.dBFS, 1)
            verbose("Movie sound fragment and lector voice dBFS:", fragDB, lectorDB)
            if fEnd <= sentence.start.ordinal:
                exitMsg("Improper movie sound fragment. Skip:", skip, "Start:", sentence.start.ordinal,
                    "stop:", fEnd, "Subtitle", sentence)
            voice = fitLectorVoice(voice, nFragment, gain)
        else:
            voice = voice.apply_gain(gain)
        lectorDB = round(voice.dBFS, 1)
        verbose("Lector voice dBFS after gain:", lectorDB)
    return voice

def audioMixFilter():
    cFilter = "[0:a:0]" if args.audioTrackIdx is None else "[0:" + str(args.audioTrackIdx) + "]"
    cFilter += "[1:a]amix=inputs=2:normalize=0:duration="
    cFilter += "shortest" if args.maxSubtitles != 0 else "longest"
    return cFilter

def mix2audio():
    car = ["ffmpeg", "-hide_banner", "-y", "-i", args.nativeMedia, "-i", tmp_merge]
    cFilter = audioMixFilter()
    car += ["-filter_complex", cFilter]
    duration = int(len(nativeSpeech) / 1000)
    if args.nativeMediaType == 'v':
        car += ["-vn", "-sn", "-an"]
    if args.maxSubtitles != 0:
        global lector
        littleExpand = 5000  # in milliseconds
        if len(nativeSpeech) - len(lector) > littleExpand:
            lector = lector.append(AdSeg.silent(littleExpand, defFrameRate), crossfade=0)
        duration = int(len(lector) / 1000)
        car += ['-to', str(duration)]
        quiet("Audio truncated due to specified --maxSubtitles option!")
    audioCodec = args.nativeStreams[args.audioTrackIdx].codec_name
    if args.merge == 'a' and (audioCodec != 'eac3' or not args.outFileDeduced):
        toFile = args.outFile
    else:
        toFile = tmp_2audios + audioCodec
        car += ["-c:a", audioCodec]
    car.append(toFile)
    quiet("Merge voice-over '" + tmp_merge + "'" + " with original '" + args.nativeMedia + "' audio track")
    cmdOutput(car, progressTime=duration)
    if args.merge == 'a' and args.outFileDeduced and audioCodec == 'eac3':
        shutil.move(toFile, args.outFile)
        args.outFile = toFile
    return toFile

def mp4merge(mFile: str):
    sms = ""
    msg = " video"
    dur = ''
    if args.maxSubtitles != 0:
        dur = ':dur=' + str(int(len(lector) / 1000))
    car = ["mp4box", "-add", args.nativeMedia + dur, "-add", mFile + ':lang=' + args.language + dur]
    if 's' in args.merge:
        sms = " and subtitle"
        car += ["-add", args.srtFile + ':lang=' + args.language + dur]
    car += ["-new", args.outFile]
    end = os.linesep if verboseFlag else ""
    quiet("Merge voice-over '" + mFile + "'" + sms + " with original '" + args.nativeMedia + "'" + msg + ' ... ', end=end)
    cmdOutput(car)
    quiet('Done')

def mkvMerge(mFile: str):
    car = ["mkvmerge", "-o", args.outFile, args.nativeMedia, "--language", '0:' + args.language, mFile]
    sms = ""
    if 's' in args.merge:
        sms = " and subtitle"
        car += ['0:' + args.language, args.srtFile]
    msg = " video"
    duration = 100
    car.append("--gui-mode")
    if args.maxSubtitles != 0:
        duration = int(len(lector) / 1000)
        car += ['--split', "parts:0s-" + str(duration) + "s"]
        duration = int((len(lector) / len(nativeSpeech)) * 100)
    quiet("Merge voice-over '" + mFile + "'" + sms + " with original '" + args.nativeMedia + "'" + msg)
    cmdOutput(car, progressTime=duration)

def mergeWithMedia():
    destFile = mix2audio()
    if destFile != args.outFile:
        # Add track to native video media
        if Path(args.outFile).suffix == '.mkv':
            mkvMerge(destFile)
            return
        elif Path(args.outFile).suffix == '.mp4':
            mp4merge(destFile)
            return
        print("WARNING! Running 'ffmpeg' to add audio file to video container is uncertain. Sometime produce corrupted audio track in movie.")
        print("Output files .mkv and .mp4 are supported natively (via mkvmerge and mp4box).")
        duration = int(len(nativeSpeech) / 1000)
        car = ["ffmpeg", "-hide_banner", "-y"]
        msg = ""
        end = os.linesep
        if args.maxSubtitles != 0:
            duration = int(len(lector) / 1000)
            car += ['-to', str(duration)]
            duration = 0
            msg = "..."
            if not args.verbose:
                end = ""
        car += ["-i", args.nativeMedia, "-i", destFile]
        sms = ""
        videoStreams = len(args.nativeStreams)
        if 's' in args.merge:
            car += ["-i", args.srtFile]
        car += ["-map", "0", "-map", "1"]
        setMetaLang(car, videoStreams)
        if 's' in args.merge:
            car += ["-map", "2"]
            setMetaLang(car, videoStreams + 1)
            sms = " and subtitle "
        # car += ["-c:v", "copy", "-c:s", "copy", "-c:a", "copy"]
        if args.maxSubtitles != 0:
            car += ['-shortest']
            quiet("Result truncated due to specified --maxSubtitles option!")
        car.append(args.outFile)
        quiet("Merge audio '" + destFile + "'" + sms + " with original '" + args.nativeMedia, msg, end=end)
        cmdOutput(car, progressTime=duration)
        if duration == 0 and not args.verbose:
            quiet(" done")

def runExternalScript():
    execFile = None
    if '-e' in sys.argv:
        execFile = sys.argv.index('-e')
    elif '--exec' in sys.argv:
        execFile = sys.argv.index('--exec')
    if execFile:
        sys.argv = sys.argv[execFile + 1:]
        if sys.argv[0].endswith('.py'):
            sys.argv[0] = sys.argv[0][:-3]
        print("Run external script:", sys.argv)
        print("Output from running script\n------------------------------------------------")
        runpy.run_module(sys.argv[0], run_name= "__main__")
        sys.exit(0)

def dumpBatchVoices(fragIdx, batch, fragments):
    print("Subtitles in batch:")
    for k in range(len(fragIdx)):
        idx = fragIdx[k]
        print(idx, ":", batch[k])
        if args.fragmentsDir is not None and len(fragments) > 0:
            fl = "split-voice-failed-" + str(idx) + ".wav"
            fl = os.path.join(args.fragmentsDir, fl)
            if k < len(fragments):
                sFrag = fragments[k]
                sFrag.export(fl, format="wav")

def genInBatch(start, stop):
    ch = 0
    batch =[]
    fragIdx = []
    while ch < args.batchSize and start < stop:
        if not fragmentPreloaded(start, 0):
            sent = getSubtitle(start)
            if not args.batchMergeMode or len(sent.text) >= 10:
                batch.append(sent.text)
                ch += len(sent.text)
                fragIdx.append(start)
        start += 1
    print("Batch generation for subtitles ", fragIdx, ch, "bytes")
    fragments = createVoice(batch)
    if len(fragments) != len(fragIdx):
        print(f"WARNING: Expected {len(fragIdx) - 1} markers, {len(fragments) - 1} detected.")
        if len(fragments) > len(fragIdx):
            print("To much fragments. Drop all generated voices")
            return start
        # print("Remove unsplit fragments")
        unsplitted = vu.findUnsplitSegments(fragments, batch, args.language)
        removeIdx = []
        removeFrags = []
        for seg in unsplitted:
            sidx = seg["subtitle_indices"]
            removeIdx.extend(sidx)
            fidx = seg["segment_index"]
            removeFrags.append(fidx)
            print("Unsplitted segment:", str(fidx) + " == " + str(fragIdx[fidx]), "Indices:", sidx)
        for index in sorted(removeIdx, reverse=True):
            del fragIdx[index]
        for index in sorted(removeFrags, reverse=True):
            del fragments[index]
        # exitMsg("Generation Failed")
    for i in range(len(fragIdx)):
        k = fragIdx[i]
        sFrag = fragments[i]
        args.voiceCache[k] = sFrag
    return start

def main():
    global args, lector
    runExternalScript()
    parser = argparse.ArgumentParser(description=usage, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-l', '--language', required=(frozen is None or ('-e' not in sys.argv and '--extract' not in sys.argv)),
            help="Subtitle language")
    parser.add_argument('-c', '--config',
            help="Config .ini file with language depended settings. Default 'srt2voice.ini'")
    parser.add_argument('-s', '--shift', help="Shift audio in milliseconds", type=int)
    parser.add_argument('-v', '--volume',
            help="Speech amplification in dB. If starts with 'r' then volume is relative to original audio volume level (-n option)")
    parser.add_argument('--highfilter', default='1000/2.5', type=str,
            help="High pass filter for male lector options in the form 'frequency Hz/compensation in dB. Default 1000 Hz/2.5 dB")
    parser.add_argument('-a', '--accelerate', help="Maximal and regular speed up voice in form maxValue:value")
    parser.add_argument('-g', '--gapAccelerate', help="Maximal and regular silence remove in form maxValue:value")
    parser.add_argument('-o', '--outFile',
            help="Output media file or directory. File name can be given without extension. It will be deduced from 'nativeMedia' type.")
    parser.add_argument('-n', '--nativeMedia', help="Original media file")
    parser.add_argument('-t', '--tracks',
            help="Specify tracks in movie to process. Tracks separated by ':'. Track format: <track type 's'/'a'><index> (e.g. -t s2:a3)")
    parser.add_argument('-m', '--merge', choices=['a', 'v', 'vs'],
            help="Merge generated audio with original audio[a] or video[v] or video+subtitle[vs] file (require -n option)")
    parser.add_argument('--formatter', default='cleanSubtitle.py',
            help="Module with defined function 'clean()'. See default formatter 'cleanSubtitle.py' for details")
    parser.add_argument('--voiceGenerator',
            help="Module with defined function 'generateVoice(text)' that returns AudioSegment or None. Built-in generator use gTTS to generate voice")
    parser.add_argument('--host', help="OpenAI TTS engine server address IP:PORT. Example for local Omnivoice-server: http://127.0.0.1:8880/v1")
    parser.add_argument('--model', help="OpenAI TTS model name. (tts-1, tts-1-hd, ...). Can be empty for Omnivoice-server")
    parser.add_argument('--key', help="OpenAI key. Can be empty for Omnivoice-server")
    parser.add_argument('-q','--quiet', action='store_true',  default=False, help="Does not display any messages while processing.")
    parser.add_argument('-k', '--keepFragments',
            required=('-p' in sys.argv or '--preloadOnly' in sys.argv or '-u' in sys.argv or '--usePreloaded' in sys.argv),
            help="Save specified fragments in separate external files. Format: [directory][:list of fragment indexes separated by ',']")
    parser.add_argument('-p','--preloadOnly', action='store_true',
            help="Preload only .wav files for specified subtitles (option -k required)")
    parser.add_argument('-u','--usePreloaded', action='store_true',
            help="Use generated .wav files for subtitles stored in keepFragment directory (option -k required)")
    parser.add_argument('--verbose', help="Print info on every subtitle and generated voice", default=0)
    parser.add_argument('--silence', help="Removes the silence between subtitle words. Format: minLength:threshold:step")
    parser.add_argument('--voice', help="Voice for language if different ones are available.")
    parser.add_argument('--maxSubtitles', help="Max subtitles for process", type=int, default=0)
    parser.add_argument('-b','--batchSize', help="Voice generation in batch. Option format: <bytes> or <bytes>/M")
    parser.add_argument('--startSubtitle', help="Start index subtitle for process [index start from 1]", type=int, default=0)
    if frozen is not None:
        parser.add_argument('-e', '--exec', help="Executes a Python script using the frozen interpreter.")
        parser.add_argument('--extract', help="Extract source files from package", nargs='?', type=str, const="*")
    parser.add_argument("srtFile", nargs='?', help="Subtitle file name")
    args = parser.parse_args()
    if frozen is not None and args.extract:
        extractSource()
    try:
        args.isoLang = iso639.Language.match(args.language)
    except iso639.LanguageNotFoundError:
        exitMsg('Language', args.language, 'not found in ISO-639 standard table')
    processArguments()
    i = args.startSubtitle
    subsLen = subsCount if args.maxSubtitles == 0 else args.maxSubtitles
    stop = i + subsLen
    lector = AdSeg.empty()
    lgt = 50
    if(subsCount <= 0):
        exitMsg("No subtitles in", args.srtFile, 'file')
    if subsCount < 100:
        formatIdx = '02d'
    elif subsCount >= 1000:
        formatIdx = '04d'
    title = "Preload speech fragments for subtitles" if args.preloadOnly else "Generate voice-over"
    charSum = 0
    generatedVoiceTime = 0
    # Do calibration of lector voice
    if args.calibrationtext is not None:
        voice = createVoice(args.calibrationtext)
        voice = removeSilence(voice)
        args.lectorTimbre = highToneRatio(voice, cutoff_freq=args.highFilterFreq)
    args.voiceCache = {} if args.batchSize > 0 else None
    batchStart = i
    with alive_bar(subsLen, title=title, length=lgt, disable=args.quiet, **barStyle) as bar:
        while i < stop:
            if args.batchSize > 0 and i >= batchStart:
                batchStart = genInBatch(batchStart, stop)
            sent = getSubtitle(i)
            charSum += len(sent.text)
            if not args.preloadOnly:
                verbose("--- Voice-over length", len(lector), "ms")
            speech, gap, skip = generateFragment(i)
            generatedVoiceTime += speech.duration_seconds * 1000
            if not args.preloadOnly:
                verbose("Generated fragment dBFS:", round(speech.dBFS, 1))
                sft = 0
                if args.shift > 0:
                    sft = max(gap - len(speech), 0)
                    if sft > 0:
                        # Is free space to shift
                        sft = min(sft, args.shift)
                        if sft > 0:
                            verbose("Generated fragment shifted by", sft, "ms")
                    else:
                        sft = 0
                l_lector = len(lector)
                sTime = sent.start.ordinal - l_lector + sft
                # skip to start of subtitle
                if sTime > 0:
                    slc = AdSeg.silent(sTime, defFrameRate)
                    lector = lector.append(slc, crossfade=0)
                    l_lector = len(lector)
                nextIdx = i + 1 if skip > 0 else 0
                speech = normalizeFragment(speech, sent, nextIdx)
                if sft > 0:
                    verbose("Append to voice-over at", len(lector), "ms")
                lector = lector.append(speech, crossfade=0)
                verbose("")
            bar()
            i += 1
            if skip > 0:
                i += 1
                if i <= stop:
                    bar()
    charSum = round(charSum / 1024, 2)
    quiet(charSum, "KB chars in subtitle file.")
    generatedVoiceTime = round(generatedVoiceTime / 1000, 2)
    quiet(generatedVoiceTime, "sec. voice-over time.")
    if not args.preloadOnly:
        if not args.merge:
            fFormat = Path(args.outFile).suffix
            eFile = args.outFile
        else:
            lector = lector.set_frame_rate(48000).set_channels(2)
            fFormat = Path(tmp_merge).suffix
            eFile = tmp_merge
        fFormat = fFormat[1:]
        quiet("Save generated audio to file:", eFile)
        if fFormat == 'aac':
            fFormat = "adts"
        lector.export(eFile, format=fFormat)
        if args.merge:
            mergeWithMedia()

if __name__ == '__main__':
    main()
