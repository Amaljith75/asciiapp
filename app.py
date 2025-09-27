import cv2
import numpy as np
import os
from flask import Flask, request, send_file, jsonify
import subprocess
import tempfile
import shutil
import uuid

app = Flask(__name__)

# Safely define BASE_DIR
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()  # Fallback to current working directory
INDEX_PATH = os.path.join(BASE_DIR, 'index.html')

# Check if index.html exists
if not os.path.exists(INDEX_PATH):
    print(f"Error: index.html not found at {INDEX_PATH}")

def convert_frame_to_ascii(frame, width=80, color_mode='grayscale', aspect_scale=0.5):
    ascii_chars = " .:-=+*#%@"  # Original character set
    height = int(frame.shape[0] * width / frame.shape[1] * aspect_scale)
    if height == 0:
        height = 1
    resized_frame = cv2.resize(frame, (width, height))
    if color_mode == 'color':
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    normalized = gray_frame / 255.0
    ascii_lines = []
    colors = [] if color_mode == 'color' else None
    for y in range(height):
        line = []
        line_colors = [] if color_mode == 'color' else None
        for x in range(width):
            pixel = normalized[y, x]
            index = int(pixel * (len(ascii_chars) - 1))
            char = ascii_chars[index]
            line.append(char)
            if color_mode == 'color':
                b, g, r = resized_frame[y, x]
                line_colors.append((int(r), int(g), int(b)))
        ascii_lines.append(''.join(line))
        if color_mode == 'color':
            colors.append(line_colors)
    return ascii_lines, colors

def render_ascii_to_image(ascii_lines, colors, font_size=8, color_mode='grayscale'):
    char_width = font_size
    line_height = int(font_size * 1.2)
    width = len(ascii_lines[0]) * char_width
    height = len(ascii_lines) * line_height
    img = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y, line in enumerate(ascii_lines):
        for x, char in enumerate(line):
            color = (255, 255, 255) if color_mode != 'color' else colors[y][x]
            cv2.putText(img, char, (x * char_width, y * line_height + line_height), font, 0.5, color, 1, cv2.LINE_AA)
    return img

@app.route('/')
def serve_index():
    try:
        print(f"Attempting to serve index.html from {INDEX_PATH}")
        return send_file(INDEX_PATH)
    except FileNotFoundError:
        print(f"Failed to serve index.html: File not found at {INDEX_PATH}")
        return jsonify({'error': 'index.html not found'}), 404

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            print("No file in request.files")
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        print(f"Processing file: {file.filename}, MIME type: {file.mimetype}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            input_path = os.path.join(temp_dir, file.filename)
            file.save(input_path)
            print(f"File saved to: {input_path}")
            
            density = int(request.form.get('density', 5))
            width = density * 20
            color_mode = request.form.get('colorMode', 'grayscale')
            font_size = int(request.form.get('fontSize', 8))
            aspect_scale = float(request.form.get('aspectRatio', 0.5))
            keep_audio = request.form.get('keepAudio', 'true') == 'true'
            volume = float(request.form.get('volume', 0.8))
            start_time = request.form.get('startTime', None)
            end_time = request.form.get('endTime', None)
            
            print(f"Parameters: density={density}, color_mode={color_mode}, font_size={font_size}, aspect_scale={aspect_scale}, keep_audio={keep_audio}, volume={volume}")
            
            # Handle custom audio
            audio_path = None
            if 'audio' in request.files:
                audio_file = request.files['audio']
                if audio_file.filename != '':
                    audio_path = os.path.join(temp_dir, audio_file.filename)
                    audio_file.save(audio_path)
                    print(f"Custom audio saved to: {audio_path}")
            
            # Process image or video
            file_type = file.mimetype.split('/')[0]
            print(f"File type: {file_type}")
            if file_type == 'image':
                file.seek(0)
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print("Failed to decode image")
                    return jsonify({'error': 'Invalid image'}), 400
                ascii_lines, colors = convert_frame_to_ascii(img, width, color_mode, aspect_scale)
                ascii_img = render_ascii_to_image(ascii_lines, colors, font_size, color_mode)
                output_path = os.path.join(temp_dir, 'ascii_image.png')
                cv2.imwrite(output_path, ascii_img)
                print(f"Image output saved to: {output_path}")
                # Copy to a new file to avoid PermissionError
                final_output = os.path.join(tempfile.gettempdir(), f'ascii_image_{uuid.uuid4().hex}.png')
                shutil.copy(output_path, final_output)
                print(f"Final image output: {final_output}")
                return send_file(final_output, as_attachment=True, download_name='ascii_image.png')
            else:  # Video
                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Video FPS: {fps}, Frame count: {frame_count}")
                output_video_path = os.path.join(temp_dir, 'ascii_output.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read video")
                    return jsonify({'error': 'Invalid video'}), 400
                ascii_lines, colors = convert_frame_to_ascii(frame, width, color_mode, aspect_scale)
                frame_img = render_ascii_to_image(ascii_lines, colors, font_size, color_mode)
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_img.shape[1], frame_img.shape[0]))
                out.write(frame_img)
                
                for i in range(1, frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"End of video at frame {i}")
                        break
                    ascii_lines, colors = convert_frame_to_ascii(frame, width, color_mode, aspect_scale)
                    frame_img = render_ascii_to_image(ascii_lines, colors, font_size, color_mode)
                    out.write(frame_img)
                
                cap.release()
                out.release()
                print(f"Video output saved to: {output_video_path}")
                
                # Check if FFmpeg is available
                if shutil.which('ffmpeg') is None:
                    print("FFmpeg not found in system PATH")
                    return jsonify({'error': 'FFmpeg is not installed on the server'}), 500
                
                # Handle audio with FFmpeg
                final_output_path = os.path.join(temp_dir, 'final_ascii.mp4')
                audio_cmd = ['ffmpeg', '-i', output_video_path, '-y']
                if audio_path or keep_audio:
                    audio_source = audio_path if audio_path else input_path
                    audio_cmd.extend(['-i', audio_source])
                    if start_time:
                        audio_cmd.extend(['-ss', start_time])
                    if end_time:
                        audio_cmd.extend(['-to', end_time])
                    audio_cmd.extend(['-filter:a', f'volume={volume}'])
                    audio_cmd.extend(['-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0'])
                else:
                    audio_cmd.extend(['-c', 'copy'])
                audio_cmd.append(final_output_path)
                print(f"FFmpeg command: {' '.join(audio_cmd)}")
                try:
                    result = subprocess.run(audio_cmd, check=True, capture_output=True, text=True)
                    print(f"FFmpeg output: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg error: {e.stderr}")
                    return jsonify({'error': 'Audio processing failed: ' + e.stderr}), 500
                
                # Copy to a new file to avoid PermissionError
                final_output = os.path.join(tempfile.gettempdir(), f'ascii_video_{uuid.uuid4().hex}.mp4')
                shutil.copy(final_output_path, final_output)
                print(f"Final video output: {final_output}")
                return send_file(final_output, as_attachment=True, download_name='ascii_video.mp4')
        finally:
            # Ensure cleanup
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Cleanup error: {e}")
    except Exception as e:
        print(f"Unexpected error in /upload: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
