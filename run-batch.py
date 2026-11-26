#!/usr/bin/env python3
# import os
import glob
import os
import subprocess
import sys
import shlex

# file name fragments tokens: %file, %dir , %name, %basename, %ext

def runProg():
    if not quiet:
        print('\nRun batch entry:', ' '.join(runArgs), "\n---------------------------------------------------------\n")
    subprocess.run(runArgs, stderr=sys.stderr, stdout=sys.stdout, universal_newlines=True, bufsize=1)

usage = \
"USAGE: Run external program multiple times. Options: [-q/--quiet] <control file>\n\n"\
"Control file options format:\n"\
"First (non comment line) or lines started with '~' specify <program> and global option for every run.\n"\
"Line started with '<' define mask (glob) for file selection\n"\
"In program line you can use markers (%file, %dir, %name, %basename, %ext) to build output file path if program produce some.\n"\
"Where %file=%dir/%basename and %basename=%name.%ext\n"\
"Rest lines specify options specify for individual run\n"\
"Comments lines (#) are ignored"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(usage)
        print("\nERROR: Specify control file with batch options")
        sys.exit(1)
    gArgs = None
    quiet = '-q' in sys.argv or '--quiet' in sys.argv
    with open(sys.argv[1]) as fp:
        for line in fp:
            line = line.removesuffix('\n')
            if line.startswith('<'):
                files = glob.glob(line[1:])
                if len(files) != 0:
                    if gArgs is None:
                        print("Program not specified in control file")
                        sys.exit(0)
                    for file in files:
                        fdir, basename = os.path.split(file)
                        fname, ext = os.path.splitext(basename)
                        runArgs = gArgs.copy()
                        for i in range(len(runArgs)):
                            runArgs[i] = runArgs[i].replace('%basename', basename)
                            runArgs[i] = runArgs[i].replace('%dir', fdir)
                            runArgs[i] = runArgs[i].replace('%name', fname)
                            runArgs[i] = runArgs[i].replace('%ext', ext)
                            runArgs[i] = runArgs[i].replace('%file', file)
                        runProg()
                continue
            line, sep, tail = line.partition('#')
            car = shlex.split(line, posix=True)
            if len(car) == 0:
                continue
            if gArgs is None or car[0].startswith('~'):
                if car[0].startswith('~'):
                    car[0] = car[0][1:]
                tenv = []
                for i in range(len(car)):
                    env = car[i].split('=')
                    if len(env) != 2:
                        break
                    tenv.append('env')
                    tenv.append(car[i])
                gArgs = tenv + car[i:]
                continue
            # Line with additional (specific to single run) option
            runArgs = gArgs.copy()
            runArgs.extend(car)
            runProg()
    # print("\nExit program\n")
