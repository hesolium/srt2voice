import os
import re
import pysrt
from pysrt import SubRipItem
CLEANER1 = re.compile('<.*?>')
CLEANER2 = re.compile('{.*?}')
timeGap = 10    # in ms
global args

"""
It's possible to generate subtitle file or convert from other formats. 
In that case, set the args.srtFile and args.clearSrt variables to the names of the generated and cleaned files respectively (in SRT format).
Files should be in args.tempDir directory with names relative to output file name (args.outFile)
It allow parallel processing multiple movies (series) with the same keepFragment directory
"""

def cleanItem(subTitle: SubRipItem):
    sv = False
    if '<' in subTitle.text:
        sv = True
        subTitle.text = re.sub(CLEANER1, '', subTitle.text)
    if '{' in subTitle.text:
        sv = True
        subTitle.text = re.sub(CLEANER2, '', subTitle.text)
    return sv

if args.srtFile is not None:
    save = False
    subs = pysrt.open(args.srtFile)
    i = 0
    while i < len(subs):
        sb:SubRipItem = subs[i]
        if i < len(subs) - 1:
            # check and remove duplicates (near the same start time)
            if subs[i + 1].start.ordinal - sb.start.ordinal <= timeGap:
                subs.remove(subs[i + 1])
                save = True
        if cleanItem(sb):
            save = True
        i += 1
    if save:
        args.clearSrt = os.path.join(args.tempDir, args.outFile + "-tmpClean.srt")
        subs.save(args.clearSrt)
