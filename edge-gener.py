import asyncio
import aiohttp
import io
import time
import edge_tts
from pydub import AudioSegment
from argparse import Namespace

args: Namespace = None

_LOOP = None

engineName = 'edge'

def _get_event_loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
    return _LOOP

async def _generate_edge_stream(text: str) -> io.BytesIO:
    communicate = edge_tts.Communicate(text, args.voice, pitch=args.pitch)
    mp3_fp = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_fp.write(chunk["data"])
    mp3_fp.seek(0)
    return mp3_fp

def generateVoice(text: str) -> AudioSegment | None:
    global args
    if isinstance(text, list):
        text = text[0]
    if text is None or text.strip() == "":
        return None
    if not hasattr(args, "pitch") or args.pitch is None:
        args.pitch = "+0Hz"
    loop = _get_event_loop()
    maxTry = 3
    while maxTry > 0:
        try:
            mp3_fp = loop.run_until_complete(_generate_edge_stream(text))
            if mp3_fp.getbuffer().nbytes > 0:
                segment = AudioSegment.from_mp3(mp3_fp)
                return segment
            # empty stream, repeat
            maxTry -= 1
            # print("Generate failed. maxTry", maxTry)
        except Exception as e:
            maxTry -= 1
            # print("Exception. Generate failed. maxTry", maxTry)
        if maxTry > 0 and not args.quiet:
            print("Network error! Sleep for 5 seconds and try again ...")
            time.sleep(5)
    import sys
    sys.exit(1)

def init(progArgs):
    global args
    args = progArgs
