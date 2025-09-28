import cv2
import numpy as np
import os
from flask import Flask, request, send_file, jsonify
import subprocess
import tempfile
import shutil
import uuid
import math

app = Flask(__name__)

# Safely define BASE_DIR
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
INDEX_PATH = os.path.join(BASE_DIR, 'index.html')

# Check if index.html exists
if not os.path.exists(INDEX_PATH):
    print(f"Error: index.html not found at {INDEX_PATH}")

def adjust_image(frame, brightness=0, contrast=1.0, ambience=0, clarity=0):
    # Apply brightness and contrast
    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness * 100)
    
    # Apply clarity (sharpness)
    if clarity != 0:
        blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
        adjusted = cv2.addWeighted(adjusted, 1 + clarity, blurred, -clarity, 0)
    
    # Apply ambience (soft color overlay)
    if ambience != 0:
        amb_color = np.full_like(adjusted, (50, 50, 50))
        adjusted = cv2.addWeighted(adjusted, 1 - ambience, amb_color, ambience, 0)
        
    return adjusted

def convert_frame_to_ascii(frame, width=80, color_mode='grayscale', aspect_scale=0.5, custom_symbols=" .';-+[*%", brightness=0, contrast=1.0, ambience=0, clarity=0, crop_x=0, crop_y=0, crop_width=None, crop_height=None):
    ascii_chars = custom_symbols or " .';-+[*%"
    frame = adjust_image(frame, brightness, contrast, ambience, clarity)
    
    # Apply cropping
    if crop_width and crop_height:
        frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("Invalid crop dimensions")
    
    height = int(frame.shape[0] * width / frame.shape[1] * aspect_scale)
    if height == 0:
        height = 1
    if width == 0:
        width = 1
    resized_frame = cv2.resize(frame, (width, height))
    if color_mode in ['color', 'matrix', 'matrix_red', 'matrix_violet', 'matrix_blue', 'matrix_cyan', 'matrix_heliotrope']:
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    normalized = gray_frame / 255.0
    ascii_lines = []
    colors = [] if color_mode in ['color', 'matrix', 'matrix_red', 'matrix_violet', 'matrix_blue', 'matrix_cyan', 'matrix_heliotrope'] else None
    for y in range(height):
        line = []
        line_colors = [] if colors is not None else None
        for x in range(width):
            pixel = normalized[y, x]
            index = int(pixel * (len(ascii_chars) - 1))
            char = ascii_chars[index]
            line.append(char)
            if colors is not None:
                if color_mode == 'color':
                    b, g, r = resized_frame[y, x]
                    line_colors.append((int(r), int(g), int(b)))
                elif color_mode == 'matrix':
                    line_colors.append((0, 255, 0))
                elif color_mode == 'matrix_red':
                    line_colors.append((255, 0, 0))
                elif color_mode == 'matrix_violet':
                    line_colors.append((128, 0, 128))
                elif color_mode == 'matrix_blue':
                    line_colors.append((0, 0, 255))
                elif color_mode == 'matrix_cyan':
                    line_colors.append((0, 255, 255))
                elif color_mode == 'matrix_heliotrope':
                    line_colors.append((223, 115, 255))
        ascii_lines.append(''.join(line))
        if colors is not None:
            colors.append(line_colors)
    return ascii_lines, colors

def render_ascii_to_image(ascii_lines, colors, font_size=8, color_mode='grayscale', resolution='1080p'):
    char_width = font_size
    line_height = int(font_size * 1.2)
    width = len(ascii_lines[0]) * char_width
    height = len(ascii_lines) * line_height
    img = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y, line in enumerate(ascii_lines):
        for x, char in enumerate(line):
            color = (255, 255, 255) if color_mode in ['grayscale', 'monochrome'] else colors[y][x] if colors else (255, 255, 255)
            cv2.putText(img, char, (x * char_width, y * line_height + line_height), font, 0.5, color, 1, cv2.LINE_AA)
    
    resolutions = {
        '1920p': (1920, 1080),
        '1366p': (1366, 768),
        '1080p': (1920, 1080),
        '720p': (1280, 720),
        '360p': (640, 360)
    }
    target_size = resolutions.get(resolution, (1920, 1080))
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
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
        
        file_size = len(file.read())
        file.seek(0)
        if file_size > 80 * 1024 * 1024:
            return jsonify({'error': 'File size exceeds 80MB limit'}), 400
        
        print(f"Processing file: {file.filename}, MIME type: {file.mimetype}, Size: {file_size} bytes")
        
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
            custom_symbols = request.form.get('customSymbols', " .';-+[*%")
            brightness = float(request.form.get('brightness', 0))
            contrast = float(request.form.get('contrast', 1.0))
            ambience = float(request.form.get('ambience', 0))
            clarity = float(request.form.get('clarity', 0))
            resolution = request.form.get('resolution', '1080p')
            keep_audio = request.form.get('keepAudio', 'true') == 'true'
            volume = float(request.form.get('volume', 0.8))
            start_time = request.form.get('startTime', None)
            end_time = request.form.get('endTime', None)
            crop_x = int(request.form.get('cropX', 0))
            crop_y = int(request.form.get('cropY', 0))
            crop_width = int(request.form.get('cropWidth', 0)) or None
            crop_height = int(request.form.get('cropHeight', 0)) or None
            
            print(f"Parameters: density={density}, color_mode={color_mode}, font_size={font_size}, aspect_scale={aspect_scale}, custom_symbols={custom_symbols}, brightness={brightness}, contrast={contrast}, ambience={ambience}, clarity={clarity}, resolution={resolution}, keep_audio={keep_audio}, volume={volume}, crop_x={crop_x}, crop_y={crop_y}, crop_width={crop_width}, crop_height={crop_height}")
            
            audio_path = None
            if 'audio' in request.files:
                audio_file = request.files['audio']
                if audio_file.filename != '':
                    audio_path = os.path.join(temp_dir, audio_file.filename)
                    audio_file.save(audio_path)
                    print(f"Custom audio saved to: {audio_path}")
            
            file_type = file.mimetype.split('/')[0]
            print(f"File type: {file_type}")
            if file_type == 'image':
                file.seek(0)
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print("Failed to decode image")
                    return jsonify({'error': 'Invalid image'}), 400
                ascii_lines, colors = convert_frame_to_ascii(img, width, color_mode, aspect_scale, custom_symbols, brightness, contrast, ambience, clarity, crop_x, crop_y, crop_width, crop_height)
                ascii_img = render_ascii_to_image(ascii_lines, colors, font_size, color_mode, resolution)
                output_path = os.path.join(temp_dir, 'ascii_image.png')
                cv2.imwrite(output_path, ascii_img)
                print(f"Image output saved to: {output_path}")
                final_output = os.path.join(tempfile.gettempdir(), f'ascii_image_{uuid.uuid4().hex}.png')
                shutil.copy(output_path, final_output)
                print(f"Final image output: {final_output}")
                return send_file(final_output, as_attachment=True, download_name='ascii_image.png')
            else:
                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                if duration > 60:
                    print("Video duration exceeds 1 minute limit")
                    return jsonify({'error': 'Video duration exceeds 1 minute limit'}), 400
                print(f"Video FPS: {fps}, Frame count: {frame_count}, Duration: {duration} seconds")
                output_video_path = os.path.join(temp_dir, 'ascii_output.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read video")
                    return jsonify({'error': 'Invalid video'}), 400
                ascii_lines, colors = convert_frame_to_ascii(frame, width, color_mode, aspect_scale, custom_symbols, brightness, contrast, ambience, clarity, crop_x, crop_y, crop_width, crop_height)
                frame_img = render_ascii_to_image(ascii_lines, colors, font_size, color_mode, resolution)
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_img.shape[1], frame_img.shape[0]))
                out.write(frame_img)
                
                for i in range(1, frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"End of video at frame {i}")
                        break
                    ascii_lines, colors = convert_frame_to_ascii(frame, width, color_mode, aspect_scale, custom_symbols, brightness, contrast, ambience, clarity, crop_x, crop_y, crop_width, crop_height)
                    frame_img = render_ascii_to_image(ascii_lines, colors, font_size, color_mode, resolution)
                    out.write(frame_img)
                
                cap.release()
                out.release()
                print(f"Video output saved to: {output_video_path}")
                
                if shutil.which('ffmpeg') is None:
                    print("FFmpeg not found in system PATH")
                    return jsonify({'error': 'FFmpeg is not installed on the server'}), 500
                
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
                
                final_output = os.path.join(tempfile.gettempdir(), f'ascii_video_{uuid.uuid4().hex}.mp4')
                shutil.copy(final_output_path, final_output)
                print(f"Final video output: {final_output}")
                return send_file(final_output, as_attachment=True, download_name='ascii_video.mp4')
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Cleanup error: {e}")
    except Exception as e:
        print(f"Unexpected error in /upload: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
