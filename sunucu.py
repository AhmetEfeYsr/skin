from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_skin_tone(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Resim yüklenemedi"

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    skin_pixels = []

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None, "Yüz bulunamadı"

        # İlk yüzü al
        face_landmarks = results.multi_face_landmarks[0]

        # Yanak ve alın noktaları (face mesh point index'lerine göre)
        sample_points = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]  # alın + yanak bölgeleri gibi

        for idx in sample_points:
            pt = face_landmarks.landmark[idx]
            x = int(pt.x * w)
            y = int(pt.y * h)
            if 0 <= x < w and 0 <= y < h:
                skin_pixels.append(rgb_image[y, x])

        if not skin_pixels:
            return None, "Cilt örnekleri alınamadı"

        avg_color = np.mean(skin_pixels, axis=0).astype(int)
        r, g, b = avg_color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        # Sınıflandırma
        if r > 200 and g > 180 and b > 150:
            color_name = "Açık Ten"
        elif r > 170 and g > 140 and b > 110:
            color_name = "Orta Açık Bej"
        elif r > 140 and g > 110 and b > 90:
            color_name = "Buğday"
        elif r > 110 and g > 90 and b > 70:
            color_name = "Koyu Buğday"
        else:
            color_name = "Koyu Ten"

        return hex_color, color_name

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'photo' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "Dosya seçilmedi"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        hex_color, color_name = get_skin_tone(filepath)

        try:
            os.remove(filepath)
        except:
            pass

        if hex_color is None:
            return jsonify({"error": color_name}), 400

        return jsonify({
            "dominant_color": hex_color,
            "color_name": color_name
        })

    return jsonify({"error": "Geçersiz dosya formatı"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
