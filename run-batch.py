#!/usr/bin/env python3
# import os
import os
import subprocess
import sys
import shlex

usage = \
"USAGE: Run external program multiple times. Run options specified in file\n\n"\
"Run file options format:\n"\
"First (non comment line) or lines started with ~ specify <program> and global option for every run\n"\
"Rest lines specify options for individual run\n"\
"Comments lines are ignored"

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(usage)
        print("\nERROR: Specify file with batch options")
        sys.exit(1)
    gArgs = None
    quiet = '-q' in sys.argv or '--quiet' in sys.argv
    with open(sys.argv[1]) as fp:
        for line in fp:
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
            runArgs = gArgs.copy()
            i = 0
            while i < len(car):
                if car[i].startswith('-'):
                    # replace or append option
                    k = runArgs.index(car[i]) if car[i] in runArgs else -1
                    if i < len(car) - 1 and not car[i + 1].startswith('-'):
                        # option with value
                        if k < 0:
                            # append
                            runArgs += [car[i], car[i + 1]]
                            i += 1
                        else:
                            # replace value
                            runArgs[k + 1] = car[i + 1]
                    else:
                        runArgs.append(car[i])
                else:
                    runArgs.append(car[i])
                i += 1
            if not quiet:
                print('\nRun batch entry:', ' '.join(runArgs), "\n--------------------------------\n")
            subprocess.run(runArgs, stderr=sys.stderr, stdout=sys.stdout, universal_newlines=True, bufsize=1)
