import os
import cv2
import re
import numpy as np
import pandas as pd
import yaml
import onnxruntime as ort
from ultralytics import YOLO
import Levenshtein
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD MODEL (YOLO11s + KUSTOM CCT OCR)
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))

yolo_absolute_path = os.path.join(base_dir, "best.pt")
print(f"🔍 Memuat file YOLO dari jalur: {yolo_absolute_path}")
model = YOLO(yolo_absolute_path)

onnx_path = os.path.join(base_dir, "cct_xs_v1_global.onnx")
config_path = os.path.join(base_dir, "config", "indonesian_plate_config.yaml")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
alphabet = config['alphabet']

ocr_session = ort.InferenceSession(onnx_path)
ocr_input_name = ocr_session.get_inputs()[0].name

# ==========================================
# 2. PATH JALUR FOLDER UJI COBA
# ==========================================
IMAGE_FOLDER = os.path.join(base_dir, "uji_coba")
HASIL_FOLDER = os.path.join(base_dir, "hasil")
os.makedirs(HASIL_FOLDER, exist_ok=True)

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper()).strip()

# 💡 PERBAIKAN: Definisikan semua variabel akumulator di sini sebelum loop dimulai
total_images = 0
global_correct_words = 0
global_total_chars = 0
global_correct_chars = 0
hasil_list = []
all_y_true = []  # <--- Ditambahkan agar tidak NameError lagi
all_y_pred = []  # <--- Ditambahkan agar tidak NameError lagi

print("\n" + "="*75)
print("--- AUTOMATIC PAIRED-LABEL PIPELINE EVALUATION (YOLO + OCR) ---")
print("="*75)

if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ Error: Folder '{IMAGE_FOLDER}' tidak ditemukan!")
else:
    # Ambil semua file gambar di folder uji_coba
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    daftar_foto = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(valid_extensions)]

    if len(daftar_foto) == 0:
        print(f"⚠️ Folder '{IMAGE_FOLDER}' kosong! Masukkan foto beserta file .txt labelnya.")
    else:
        # Proses iterasi otomatis untuk setiap foto
        for idx, file_name in enumerate(daftar_foto, 1):
            image_path = os.path.join(IMAGE_FOLDER, file_name)
            
            # Cari file .txt pasangannya (misal: test001.jpg -> test001.txt)
            nama_tanpa_ekstensi, _ = os.path.splitext(file_name)
            txt_label_path = os.path.join(IMAGE_FOLDER, f"{nama_tanpa_ekstensi}.txt")
            
            # Cek apakah file .txt labelnya ada atau tidak
            if not os.path.exists(txt_label_path):
                print(f"⚠️ [{idx}] File label '{nama_tanpa_ekstensi}.txt' tidak ditemukan. Dilewati.")
                continue

            # Baca isi file .txt untuk mengambil plat nomor asli (Ground Truth)
            with open(txt_label_path, 'r') as f:
                isi_label = f.read().strip()
                if not isi_label:
                    continue
                parts = isi_label.split()
                plat_asli = parts[-1] 
            
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                continue

            ground_truth = clean_text(plat_asli)
            total_images += 1

            # --- TAHAP 1: DETEKSI YOLO ---
            img_rgb_yolo = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = model(img_rgb_yolo, verbose=False)
            boxes = results[0].boxes

            if len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate = img_bgr[y1:y2, x1:x2]
                status_yolo = "SUKSES DETEKSI"
            else:
                plate = img_bgr
                status_yolo = "GAGAL DETEKSI (Pakai Full Gambar)"

            # --- TAHAP 2: BACA OCR ---
            pred_text = ""
            if plate.size > 0:
                img_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, (128, 64))
                img_rgb_ocr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                img_input = np.expand_dims(img_rgb_ocr, axis=0)

                preds = ocr_session.run(None, {ocr_input_name: img_input})[0]
                best_path = np.argmax(preds, axis=-1)[0]
                
                raw_text = "".join([alphabet[i] for i in best_path if i < len(alphabet) and alphabet[i] != '_'])
                pred_text = clean_text(raw_text)

            # --- PERHITUNGAN AKURASI PER PLAT ---
            is_word_correct = (pred_text == ground_truth)
            if is_word_correct:
                global_correct_words += 1
            word_acc_per_plat = 100.0 if is_word_correct else 0.0

            distance = Levenshtein.distance(ground_truth, pred_text)
            max_len = max(len(ground_truth), len(pred_text))
            correct_chars_per_plat = max_len - distance if max_len > 0 else 0
            
            char_acc_per_plat = (correct_chars_per_plat / max_len) * 100 if max_len > 0 else 0.0

            global_total_chars += max_len
            global_correct_chars += correct_chars_per_plat

            # KUMPULKAN DATA UNTUK CONFUSION MATRIX
            for i in range(min(len(ground_truth), len(pred_text))):
                all_y_true.append(ground_truth[i])
                all_y_pred.append(pred_text[i])

            # Simpan rekap data
            hasil_list.append({
                "Nama File": file_name,
                "Plat Asli": ground_truth,
                "Prediksi AI": pred_text,
                "YOLO Status": status_yolo,
                "Word Accuracy": f"{word_acc_per_plat:.2f}%",
                "Char Accuracy": f"{char_acc_per_plat:.2f}%"
            })

            # Cetak rincian per plat ke terminal
            print(f"[{idx}/{len(daftar_foto)}] File: {file_name}")
            print(f"      🔹 Plat Asli (GT)  : {ground_truth}")
            print(f"      🔹 Prediksi AI     : {pred_text}")
            print(f"      🔹 Word Accuracy   : {word_acc_per_plat:.2f}%")
            print(f"      🔹 Char Accuracy   : {char_acc_per_plat:.2f}%")
            print("-" * 65)

        # ==========================================
        # 3. REKAPITULASI TOTAL KESELURUHAN & CM
        # ==========================================
        if total_images > 0:
            global_word_accuracy = (global_correct_words / total_images) * 100
            global_char_accuracy = (global_correct_chars / global_total_chars) * 100

            print("\n📊 " + "="*23 + " REKAP AKURASI KESELURUHAN " + "="*23)
            print(f" Total Data Sukses Teruji           : {total_images} Kendaraan")
            print(f" Total Word Accuracy Keseluruhan    : {global_word_accuracy:.2f}%")
            print(f" Total Character Accuracy Keseluruhan: {global_char_accuracy:.2f}%")
            print("="*75)

            # Simpan data ke CSV
            df = pd.DataFrame(hasil_list)
            csv_path = os.path.join(HASIL_FOLDER, "rekap_evaluasi_paired_label.csv")
            df.to_csv(csv_path, index=False)
            print(f"💾 [INFO] Hasil detail sukses diexport ke CSV: {csv_path}")

            # 4. GENERATE GRAFIK CONFUSION MATRIX TINGKAT KARAKTER
            if len(all_y_true) > 0:
                labels_cm = sorted(list(set(all_y_true + all_y_pred)))
                cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_cm)
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=labels_cm, yticklabels=labels_cm,
                            annot_kws={"size": 9})
                
                plt.xlabel("Predicted Characters (Prediksi Sistem AI)", fontsize=11, fontweight='bold')
                plt.ylabel("Ground Truth Characters (Label Plat Asli)", fontsize=11, fontweight='bold')
                plt.title("Confusion Matrix Karakter - Integrasi Pipeline YOLO + OCR", fontsize=13, pad=15, fontweight='bold')
                
                plt.tight_layout()
                
                path_cm_gambar = os.path.join(HASIL_FOLDER, "confusion_matrix_karakter.png")
                plt.savefig(path_cm_gambar, dpi=300)
                plt.close()
                print(f"📈 [INFO] Grafik Confusion Matrix sukses disimpan ke: {path_cm_gambar}\n")
        else:
            print("\n❌ Tidak ada gambar yang memiliki file .txt label valid untuk dievaluasi!")