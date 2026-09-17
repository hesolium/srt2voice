import os
import sys
import time
from aiohttp.client_exceptions import ClientConnectorError
import edge_tts
global args # srt2voice module set this variable after load THIS module

def genVoice(text, outFile):
    try:
        communicate = edge_tts.Communicate(text, args.voice, pitch=args.pitch)
        communicate.save_sync(outFile)
        rv = True
    except OSError:
        if os.path.isfile(outFile):
            os.remove(outFile)
        rv = False
    return rv

def generateVoice(text, outFile):
    if not hasattr(args, "pitch") or args.pitch is None:
        args.pitch = "+0Hz"
    if os.path.isfile(outFile):
        os.remove(outFile)
    maxTry = 3
    while maxTry > 0:
        if genVoice(text, outFile):
            break
        if not args.quiet:
            print("Network error! Sleep for 5 seconds and try again...")
        time.sleep(5)
        maxTry -= 1
        if maxTry == 0:
            print("Generating voice for '" + text + "' failed")
            sys.exit(1)
