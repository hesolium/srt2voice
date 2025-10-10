py -3.12 -m PyInstaller --name srt2voice --onefile ^
    --add-data srt2voice.ini:. --add-data requirements.txt:. --add-data README:. --add-data srt2voice.py:. --add-data make-release.bat:. ^
    --add-data run-batch.py:. --add-data cleanSubtitle.py:. --add-data edge-gener.py:. --add-binary audio-stretch.exe:. --add-binary ffmpeg/bin:. ^
    --add-binary mkvmerge.exe:. --add-binary mp4box.exe:. srt2voice.py
