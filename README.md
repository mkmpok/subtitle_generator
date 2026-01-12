# subtitle_generator

########## Create virtual environment (very recommended) 

Run these commands one by one:

Windows (cmd or PowerShell)
Bashpython -m venv venv
venv\Scripts\activate


- Mac / Linux
Bashpython3 -m venv venv
source venv/bin/activate

You should now see (venv) at the beginning of the line.

########  Install required packages
Run this command (copy-paste the whole line):
-
pip install streamlit moviepy openai-whisper opencv-python numpy pillow


Install ImageMagick (very important – needed for text in moviepy)
Windows

Download from: https://imagemagick.org/script/download.php
→ Choose: ImageMagick-7.1.1-39-Q16-HDRI-x64-dll.exe (or newest Q16-HDRI version)
Install it → default path is fine (usually C:\Program Files\ImageMagick-7.x.x-Q16-HDRI)
Important: during installation → check "Install legacy utilities (e.g. convert)"

############

 - Mac
Bashbrew install imagemagick

Run the app:

- streamlit run app.py

A browser window should open automatically: http://localhost:8501
Now you can:

Upload a video
Click "Generate Word-Level Subtitles"
Edit text if needed
Click "Generate FAST Video"
'''
