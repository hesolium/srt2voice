py -3.12 -m PyInstaller --name srt2voice --onefile ^
    --add-data srt2voice.ini:. --add-data requirements.txt:. --add-data README:. --add-data srt2voice.py:. --add-data make-release.bat:. ^
    --add-data run-batch.py:. --add-data cleanSubtitle.py:. --add-data edge-gener.py:. --add-data omni-patterns;omni-patterns ^
    --add-binary mp4box.exe:. --add-data omni-gener.py:. --add-data openai_gener.py:. --add-data start-omniserver.py:. srt2voice.py
