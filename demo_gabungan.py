import os
import cv2
import re
import numpy as np
import yaml
import onnxruntime as ort
from ultralytics import YOLO

# ==========================================
# 1. INSULASI JALUR MODEL
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(base_dir, "best.pt")
onnx_path = os.path.join(base_dir, "cct_xs_v1_global.onnx")
config_path = os.path.join(base_dir, "config", "indonesian_plate_config.yaml")

# 💡 PERUBAHAN DI SINI: Arahkan jalur foto ke dalam folder "uji_coba"
IMAGE_FOLDER = os.path.join(base_dir, "uji_coba")
path_foto = os.path.join(IMAGE_FOLDER, "test001.jpg")

print("\n" + "="*60)
print("--- DEMO GABUNGAN NYATA: YOLO (CROP) + OCR (READ) ---")
print("="*60)

if not os.path.exists(path_foto):
    print(f"⚠️ Error: Pastikan folder '{IMAGE_FOLDER}' sudah dibuat dan file '{os.path.basename(path_foto)}' ada di dalamnya.")
else:
    # Load Config Karakter OCR
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    alphabet = config['alphabet']

    # Load Model
    ocr_session = ort.InferenceSession(onnx_path)
    ocr_input_name = ocr_session.get_inputs()[0].name
    yolo_model = YOLO(yolo_path)

    # READ GAMBAR UTUH
    img_original = cv2.imread(path_foto)
    
    # ─── TAHAP 1: YOLO MENCARI & MEMOTONG ───
    # Gunakan format warna RGB untuk akurasi YOLO kustom
    img_rgb_yolo = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb_yolo, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("❌ YOLO gagal mendeteksi plat nomor pada foto utuh ini. Pastikan foto tidak terlalu blur.")
    else:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        print(f"🎯 [STAGE 1] YOLO mendeteksi plat! Confidence: {conf:.2f}")
        print(f"    Koordinat Kotak: [{x1}, {y1}] ke [{x2}, {y2}]")
        
        # Eksekusi Crop Otomatis oleh Sistem
        img_cropped = img_original[y1:y2, x1:x2]
        
        # Simpan hasil guntingan YOLO ke dalam folder utama untuk pembuktian visual
        cv2.imwrite(os.path.join(base_dir, "hasil_guntingan_yolo.jpg"), img_cropped)
        print("💾 [INFO] Hasil guntingan otomatis YOLO disimpan sebagai 'hasil_guntingan_yolo.jpg'")

        # ─── TAHAP 2: OCR MEMBACA HASIL GUNTINGAN ───
        print("\n🔄 [STAGE 2] Mengoper hasil guntingan otomatis ke Model OCR...")
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (128, 64))
        img_rgb_ocr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_input = np.expand_dims(img_rgb_ocr, axis=0)

        # Jalankan Sesi ONNX OCR
        preds = ocr_session.run(None, {ocr_input_name: img_input})[0]
        best_path = np.argmax(preds, axis=-1)[0]
        
        hasil_plat = "".join([alphabet[idx] for idx in best_path if idx < len(alphabet) and alphabet[idx] != '_']).strip()
        hasil_bersih = re.sub(r'[^A-Z0-9]', '', hasil_plat.upper())

        # DISPLAY HASIL AKHIR COLLABORATION
        print("\n🎉 " + "="*45)
        print(f" FOTO KENDARAAN UTUH : {os.path.basename(path_foto)}")
        print(f" TEXT PREDIKSI AI    : {hasil_bersih}")
        print("="*48 + "\n")