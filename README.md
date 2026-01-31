<img width="1911" height="798" alt="Screenshot 2026-01-31 180528" src="https://github.com/user-attachments/assets/ba1ce772-6a53-49f0-a0aa-c24f3d41ab0b" /># 🧬 Skin Health Pro+ 

Skin Health Pro+ is a **full-stack AI dermatology analysis system** that combines  
**Deep Learning (CNN)** with **professional medical APIs** to analyze skin images and generate **clinically structured medical reports**.

This project is designed for **academic, research, and portfolio demonstration purposes** and follows **responsible medical-AI practices**.

---

## 📌 Table of Contents
- Overview
- Key Features
- Supported Skin Conditions
- System Architecture
- Screenshots
- Tech Stack
- Project Structure
- Installation & Setup
- API Configuration
- Running the Application
- Model & Accuracy Notes
- Medical Disclaimer
- Limitations
- Future Enhancements
- License & Credits

---

## 🔍 Overview

Skin Health Pro+ allows users to:
1. Upload a skin image
2. Analyze it using a CNN model
3. Validate results using **Infermedica medical intelligence**
4. Display symptoms, risk factors, treatments, medicines
5. Locate nearby hospitals
6. Generate a **downloadable medical report (HTML)**

The system follows a **Human-in-the-Loop** approach — AI assists, doctors decide.

---

## ✨ Key Features

✅ CNN-based skin disease classification  
✅ Confidence-based predictions (Top-3 results)  
✅ Infermedica medical database integration  
✅ Symptom, risk factor & treatment intelligence  
✅ Medicine information with images  
✅ Nearby hospital & dermatology locator  
✅ Auto-generated **medical report (HTML)**  
✅ Modern medical-grade Streamlit UI  
✅ Fallback mechanisms (Wikipedia + internal DB)  

---

## 🧠 Supported Skin Conditions

- Eczema (Atopic Dermatitis)
- Acne
- Psoriasis
- Rosacea
- Melanoma
- Ringworm (Tinea)
- Dermatitis
- Fungal Skin Infections

---

## 🏗️ System Architecture

User Image
↓
CNN Deep Learning Model
↓
Top-3 Predictions + Confidence
↓
Infermedica Medical Validation
↓
Medical Intelligence (Symptoms, Risks, Treatment)
↓
Hospital Locator (Geoapify)
↓
Medical Report Generator (HTML)


---

## 📸 Screenshots

### 🔹 Dashboard Home
<img width="1911" height="798" alt="Screenshot 2026-01-31 180528" src="https://github.com/user-attachments/assets/d3dca952-f7e3-45af-bc7c-568c51d5883b" />

### 🔹 Image Upload & Analysis
<img width="1120" height="778" alt="Screenshot 2026-01-31 180540" src="https://github.com/user-attachments/assets/f2e513d7-eb7b-493a-a7cf-fa4f79463390" />

### 🔹 Diagnosis Result
<img width="1152" height="723" alt="Screenshot 2026-01-31 210601" src="https://github.com/user-attachments/assets/39b1c74b-8347-45ba-bf80-22fd9dd75aa3" />

### 🔹 Medical Intelligence (Infermedica)
<img width="1297" height="778" alt="Screenshot 2026-01-31 204918" src="https://github.com/user-attachments/assets/06ad6ac0-b6e8-42e6-9655-6bc9eebc23e4" />

### 🔹 Medicine Recommendations
<img width="1874" height="707" alt="Screenshot 2026-01-31 210704" src="https://github.com/user-attachments/assets/2dc31d36-e5ef-4474-abc6-383abb3cb038" />

### 🔹 Hospital Locator
<img width="783" height="791" alt="Screenshot 2026-01-31 205007" src="https://github.com/user-attachments/assets/acf5cf09-a5d0-4d59-83a2-fe10caea561b" />
<img width="786" height="533" alt="Screenshot 2026-01-31 205017" src="https://github.com/user-attachments/assets/e00ebb88-b28c-4f51-882d-e42112d9b715" />
<img width="1919" height="856" alt="Screenshot 2026-01-31 205026" src="https://github.com/user-attachments/assets/325f03ba-eec5-42b7-b50d-ced569bded40" />


### 🔹 Generated Medical Report
<img width="1919" height="860" alt="Screenshot 2026-01-31 205044" src="https://github.com/user-attachments/assets/607f4c8e-cef5-48c8-bc34-b1b907b70d3a" />
<img width="767" height="734" alt="Screenshot 2026-01-31 205119" src="https://github.com/user-attachments/assets/d9bc2b2d-1f20-4e7b-b75e-47f38877e7be" />
<img width="819" height="849" alt="Screenshot 2026-01-31 205139" src="https://github.com/user-attachments/assets/f9a52eb4-abb7-4cb0-91bc-5a8a6df56d41" />
<img width="1893" height="861" alt="Screenshot 2026-01-31 205200" src="https://github.com/user-attachments/assets/d2b9e1a6-b06c-4cd8-b8cc-4626d752ecc0" />
<img width="1900" height="861" alt="Screenshot 2026-01-31 205216" src="https://github.com/user-attachments/assets/f5f9755b-ab94-4671-a390-e14b04d5e547" />
<img width="1901" height="865" alt="Screenshot 2026-01-31 205228" src="https://github.com/user-attachments/assets/3e2eed66-0d5a-4e61-86e0-dbfc80377762" />
<img width="1904" height="862" alt="Screenshot 2026-01-31 205236" src="https://github.com/user-attachments/assets/4fe811ec-fbaa-4b01-bd41-90190f0fcaad" />

---

## 🧰 Tech Stack

| Layer | Technology |
|-----|-----------|
| Frontend | Streamlit |
| Backend | Python |
| AI Model | TensorFlow (CNN) |
| Image Processing | OpenCV, Pillow |
| ML Utilities | Scikit-Learn |
| Medical API | Infermedica |
| Location API | Geoapify |
| Data Parsing | BeautifulSoup |
| UI Animations | Lottie |

---

## 📁 Project Structure

Skin-Health-Pro/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│ └── skin_disease_model.h5
│
├── Skin diseses images/
│ └── train/
│
├── screenshots/
│ ├── dashboard.png
│ ├── upload_analysis.png
│ ├── diagnosis.png
│ ├── infermedica_data.png
│ ├── medicines.png
│ ├── hospitals.png
│ └── report.png
│
└── assets/


---

📞 Contact Developer: Sumit Lohar 📧 Email:sumitlohar063@gmail.com 🐙 GitHub: https://github.com/SumitLohar3566🔗 LinkedIn:(https://www.linkedin.com/in/sumit-lohar-498341317/)
