# app.py - FASTER VERSION (complete & fixed)
import streamlit as st
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.config import change_settings
import tempfile
import os
import numpy as np
import cv2
import re
from functools import lru_cache

# ---------------- ImageMagick Path (change if needed) ----------------
change_settings({
    "IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
    # For Linux/Mac users you might use: "/usr/bin/convert" or similar
})

# ---------------- Detectors ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

st.title("🎬 Fast Stacked Subtitles Generator")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

# ---------------- Helpers ----------------
def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")

def parse_srt(srt_text):
    entries = []
    blocks = re.split(r'\n\s*\n', srt_text.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        # Find timing line
        timing_line = None
        for line in lines:
            if '-->' in line:
                timing_line = line
                break
        if not timing_line:
            continue
        match = TIME_RE.search(timing_line)
        if not match:
            continue
        start_str, _, end_str = timing_line.partition(' --> ')
        start = sum(x * float(t) for x, t in zip([3600, 60, 1, 0.001], start_str.split(':')))
        end = sum(x * float(t) for x, t in zip([3600, 60, 1, 0.001], end_str.split(':')))
        text_lines = lines[lines.index(timing_line) + 1:]
        if text_lines:
            entries.append({
                "start": start,
                "end": end,
                "lines": text_lines
            })
    return entries

def build_initial_srt(word_groups):
    srt_lines = []
    for i, group in enumerate(word_groups, 1):
        if not group["lines"]:
            continue
        start = format_time(group["start"])
        end = format_time(group["end"])
        srt_lines.extend([str(i), f"{start} --> {end}"] + group["lines"] + [""])
    return "\n".join(srt_lines)

def words_from_segments(segments):
    words = []
    for seg in segments:
        for word_info in seg.get("words", []):
            word = word_info.get("word", "").strip()
            if word:
                words.append({
                    "word": word,
                    "start": word_info["start"],
                    "end": word_info["end"]
                })
    return words

def group_words(words, max_per_group=3):
    groups = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_per_group]
        if not chunk:
            break
        groups.append({
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "lines": [w["word"] for w in chunk]
        })
        i += max_per_group
    return groups

# ---------------- FASTER Subject Detection ----------------
@lru_cache(maxsize=128)
def detect_subject_in_segment(video_path, start_t, end_t):
    seg_dur = end_t - start_t
    if seg_dur < 0.4:
        return None

    num_samples = min(12, max(4, int(seg_dur * 8)))
    times = np.linspace(start_t, end_t - 0.001, num_samples)

    rects = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for t in times:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = []

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40,40))
        if len(faces) > 0:
            detected.extend(faces)

        profiles = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40,40))
        if len(profiles) > 0:
            detected.extend(profiles)

        if detected:
            rects.extend(detected)

    cap.release()

    if not rects:
        return None

    arr = np.array(rects, dtype=float)
    return {
        "min_x": float(np.min(arr[:, 0])),
        "max_x": float(np.max(arr[:, 0] + arr[:, 2])),
        "min_y": float(np.min(arr[:, 1])),
        "max_y": float(np.max(arr[:, 1] + arr[:, 3])),
        "cx": float(np.mean(arr[:, 0] + arr[:, 2] / 2)),
    }

# ────────────────────────────────────────────────────────────────
# Main App Logic
# ────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    # Handle new upload
    file_details = {"name": uploaded_file.name, "size": uploaded_file.size}
    if "prev_file" not in st.session_state or st.session_state.prev_file != file_details:
        st.session_state.prev_file = file_details
        for k in ["raw_groups", "edited_srt", "suggested_size"]:
            st.session_state.pop(k, None)
        if "video_path" in st.session_state:
            try:
                os.unlink(st.session_state.video_path)
            except:
                pass

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
    st.session_state.video_path = video_path

    st.video(video_path)

    # Auto-suggest font size
    if "suggested_size" not in st.session_state:
        with st.spinner("Analyzing video..."):
            vid = VideoFileClip(video_path)
            suggested = max(50, int(vid.h * 0.09))
            st.session_state.suggested_size = suggested
            vid.close()

    # Load whisper model (once)
    if "whisper_model" not in st.session_state:
        with st.spinner("Loading Whisper tiny model..."):
            st.session_state.whisper_model = whisper.load_model("tiny")

    # Generate subtitles button
    if st.button("Generate Word-Level Subtitles"):
        with st.spinner("Transcribing..."):
            result = st.session_state.whisper_model.transcribe(
                video_path,
                word_timestamps=True
            )
            all_words = words_from_segments(result["segments"])
            groups = group_words(all_words, max_per_group=3)
            st.session_state.raw_groups = groups
            st.session_state.edited_srt = build_initial_srt(groups)
            st.success("Subtitles ready! You can now edit them.")

    # ─── Editor & Style & Generate ───────────────────────────────
    if "edited_srt" in st.session_state:
        st.subheader("Edit SRT")
        edited = st.text_area(
            "Edit subtitles here (multi-line = stacked words)",
            value=st.session_state.edited_srt,
            height=300
        )
        if edited != st.session_state.edited_srt:
            st.session_state.edited_srt = edited

        st.subheader("Style Settings (faster = less thickness / no background)")
        col1, col2 = st.columns(2)
        with col1:
            font = st.selectbox("Font", ["Arial-Bold", "Impact", "Arial"], index=0)
            font_size = st.slider("Font Size", 24, 140, st.session_state.suggested_size)
            text_color = st.color_picker("Text Color", "#FFFFFF")
            outline_color = st.color_picker("Outline Color", "#000000")
            outline_width = st.slider("Outline Thickness", 1, 12, 4)

        with col2:
            margin = st.slider("Margin from edge (px)", 30, 200, 60)
            bg_opacity = st.slider("Background Opacity", 0.0, 1.0, 0.0)

        if st.button("⚡ Generate FAST Video", type="primary"):
            with st.spinner("Rendering video (fast mode)..."):
                video = VideoFileClip(video_path)
                w, h = video.size
                duration = video.duration
                clips = []

                LINE_SPACING = -int(font_size * 0.25)

                is_landscape = w > h

                try:
                    groups = parse_srt(st.session_state.edited_srt)
                except:
                    groups = st.session_state.raw_groups

                for seg in groups:
                    start = max(0.0, seg["start"])
                    end = min(seg["end"], duration)
                    if end - start < 0.25:
                        continue

                    lines = [ln.strip() for ln in seg["lines"] if ln.strip()]
                    if not lines:
                        continue

                    subject = detect_subject_in_segment(video_path, start, end)

                    padding = outline_width + 6 if bg_opacity > 0 else 0
                    max_text_w = int(w * (0.5 if is_landscape else 0.92))

                    text_clips = []
                    box_heights = []
                    max_tw = 0

                    for line in lines:
                        txt = TextClip(
                            line,
                            fontsize=font_size,
                            color=text_color,
                            font=font,
                            stroke_color=outline_color,
                            stroke_width=outline_width,
                            method='caption',
                            size=(max_text_w, None),
                            align='center'
                        )
                        tw, th = txt.size
                        max_tw = max(max_tw, tw)
                        box_heights.append(th + 2 * padding)
                        text_clips.append(txt)

                    total_height = sum(box_heights) + LINE_SPACING * max(0, len(lines)-1)
                    stack_width = max_tw + 2 * padding

                    # ─── Placement Logic ───────────────────────────────────
                    if is_landscape:
                        if subject:
                            left_dist = subject["cx"]
                            right_dist = w - subject["cx"]
                            side = "left" if left_dist > right_dist else "right"
                        else:
                            side = "left"
                        x_center = margin + stack_width/2 if side == "left" else w - margin - stack_width/2
                        y_anchor = (h - total_height) / 2
                        from_top = True
                    else:  # Portrait - prefer bottom
                        x_center = w / 2
                        if subject:
                            bottom_space = h - subject["max_y"] - margin
                            top_space = subject["min_y"] - margin
                            if bottom_space >= total_height:
                                from_top = False
                                y_anchor = h - margin
                            elif top_space >= total_height:
                                from_top = True
                                y_anchor = margin
                            else:
                                from_top = bottom_space > top_space
                                y_anchor = h - margin if not from_top else margin
                        else:
                            from_top = False
                            y_anchor = h - margin

                    # Calculate y positions
                    y_positions = [0] * len(lines)
                    if not from_top:  # from bottom
                        current_y = y_anchor
                        for i in range(len(lines)-1, -1, -1):
                            y_positions[i] = current_y - box_heights[i]
                            if i > 0:
                                current_y = y_positions[i] - LINE_SPACING
                    else:  # from top
                        current_y = y_anchor
                        for i in range(len(lines)):
                            y_positions[i] = current_y
                            if i < len(lines)-1:
                                current_y += box_heights[i] + LINE_SPACING

                    # Create final subtitle clips
                    for i, txt_clip in enumerate(text_clips):
                        tw, th = txt_clip.size
                        box_w = tw + 2 * padding
                        box_h = th + 2 * padding

                        if bg_opacity > 0:
                            bg = ColorClip(size=(box_w, box_h), color=(0,0,0)).set_opacity(bg_opacity)
                            final = CompositeVideoClip([bg.set_pos("center"), txt_clip.set_pos("center")], size=(box_w, box_h))
                        else:
                            final = txt_clip

                        final = final.set_position((x_center - box_w/2, y_positions[i]))
                        final = final.set_start(start).set_duration(end - start)
                        clips.append(final)

                    for t in text_clips:
                        t.close()

                # ─── Final Render ──────────────────────────────────────
                final_video = CompositeVideoClip([video] + clips).set_fps(video.fps)

                output_path = tempfile.mktemp(suffix="_fast_sub.mp4")

                final_video.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    bitrate="2500k",
                    preset="ultrafast",
                    threads=0,
                    logger=None,
                    ffmpeg_params=["-profile:v", "baseline"]
                )

                video.close()
                final_video.close()

                st.success("Video ready! (fast mode - quality slightly lower)")
                st.video(output_path)

                with open(output_path, "rb") as f:
                    st.download_button(
                        "📥 Download Video",
                        f,
                        file_name="fast_subtitles.mp4",
                        mime="video/mp4"
                    )

                try:
                    os.unlink(output_path)
                except:
                    pass

else:
    st.info("Please upload a video to begin.")