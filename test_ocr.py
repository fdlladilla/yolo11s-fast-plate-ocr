import os
import cv2
import numpy as np
import yaml
import onnxruntime as ort

# 1. Jalur file otomatis
base_dir = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(base_dir, "cct_xs_v1_global.onnx") 
config_path = os.path.join(base_dir, "config", "indonesian_plate_config.yaml")
path_foto = os.path.join(base_dir, "dataset_parkir", "test", "images", "test001_1.jpg")

print("\n" + "="*50)
print("--- PENGUJIAN FINAl VIA ONNX RUNTIME (DIMENSI 128 FIX) ---")
print("="*50)

if not os.path.exists(onnx_path):
    print(f"⚠️ Error: File ONNX tidak ditemukan di: {onnx_path}")
elif not os.path.exists(config_path):
    print(f"⚠️ Error: File config tidak ditemukan di: {config_path}")
elif not os.path.exists(path_foto):
    print(f"⚠️ Error: File foto tidak ditemukan di: {path_foto}")
else:
    # 2. Load alphabet dari config
    print("-> Membaca konfigurasi karakter...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    alphabet = config['alphabet']

    # 3. Load Model ONNX
    print("-> Memuat model ONNX...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    # 4. Preprocessing Gambar (Ubah lebar ke 128 sesuai permintaan ONNX)
    print("-> Memproses gambar target...")
    img = cv2.imread(path_foto)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # FIX: Diubah menjadi (128, 64) agar klop dengan Model ONNX-nya
    img_resized = cv2.resize(img_gray, (128, 64))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_input = np.expand_dims(img_rgb, axis=0)

    # 5. Prediksi menggunakan ONNX Runtime
    print("-> AI sedang membaca plat nomor...")
    preds = session.run(None, {input_name: img_input})[0]

    # 6. Decode Hasil ke Teks
    best_path = np.argmax(preds, axis=-1)[0]
    hasil_plat = "".join([alphabet[idx] for idx in best_path if idx < len(alphabet) and alphabet[idx] != '_']).strip()

    print("\n" + "🎉 " + "="*40)
    print(f"FOTO YANG DITES  : {os.path.basename(path_foto)}")
    print(f"HASIL DETEKSI AI : {hasil_plat}")
    print("="*41 + "\n")