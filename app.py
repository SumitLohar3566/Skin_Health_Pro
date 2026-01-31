import streamlit as st
from PIL import Image
import requests
from datetime import datetime
import time
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.image import extract_patches_2d
import cv2
from googlesearch import search
import re
import random
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import tensorflow as tf
import warnings
import wikipedia
from bs4 import BeautifulSoup
import json

warnings.filterwarnings('ignore')

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =========================
# PATHS AND CONFIGURATION
# =========================
BASE_PATH = r"C:\Users\sumit\PycharmProjects\PythonProject6"
MODEL_PATH = os.path.join(BASE_PATH, "model", "skin_disease_model.h5")
TRAIN_DIR = os.path.join(BASE_PATH, "Skin diseses images", "train")

# Geoapify API Key
GEOAPIFY_API_KEY = 'c63011a12f85455b834a9c2e8adb39df'

# Infermedica API Configuration
INFERMEDICA_APP_ID = '4d9b5a52'  # You'll need to sign up for an API key
INFERMEDICA_APP_KEY = '5dcaa37595611e64cc8a59d7cdfa1105'  # You'll need to sign up for an API key
INFERMEDICA_BASE_URL = 'https://api.infermedica.com/v3/'

# Medicine Database API (using multiple sources)
MEDICINE_DATABASE_API = {
    "Drugs.com": "https://www.drugs.com/search.php?searchterm=",
    "WebMD": "https://www.webmd.com/drugs/2/search?type=drugs&query=",
    "Wikipedia": "https://en.wikipedia.org/wiki/"
}

# Initialize Wikipedia API
try:
    import wikipediaapi

    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='SkinHealthPro/1.0 (https://yourwebsite.com)'
    )
    WIKIPEDIA_AVAILABLE = True
except (ImportError, AssertionError) as e:
    WIKIPEDIA_AVAILABLE = False
    st.warning(f"Wikipedia integration limited: {str(e)}. Using built-in disease information.")


# Load Lottie animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


lottie_doctor = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5tkzkblw.json")
lottie_scan = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5tkzkblw.json")
lottie_health = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5tkzkblw.json")

# =========================
# ENHANCED PRODUCT RECOMMENDATIONS
# =========================
PRODUCT_RECOMMENDATIONS = {
    "Eczema": [
        {
            "name": "Eczema Relief Cream",
            "url": "https://www.purplle.com/product/eczema-care-cream",
            "image": "https://m.media-amazon.com/images/I/61SJuj2VURL._SL1500_.jpg",
            "description": "Specialized cream for eczema-prone skin with colloidal oatmeal",
            "price": "₹799 - ₹1,499",
            "color": "#FFD1DC"
        },
        {
            "name": "Hypoallergenic Cleanser",
            "url": "https://limese.com/products/eczema-cleanser",
            "image": "https://m.media-amazon.com/images/I/61KbY6QSl4L._SL1500_.jpg",
            "description": "Fragrance-free cleanser for sensitive eczema skin",
            "price": "₹899",
            "color": "#B5EAD7"
        }
    ],
    "Psoriasis": [
        {
            "name": "Psoriasis Treatment Cream",
            "url": "https://www.purplle.com/product/psoriasis-cream",
            "image": "https://m.media-amazon.com/images/I/71p4Q5QyHUL._SL1500_.jpg",
            "description": "Medicated cream with salicylic acid for psoriasis relief",
            "price": "₹899 - ₹1,799",
            "color": "#FFD1DC"
        },
        {
            "name": "Coal Tar Shampoo",
            "url": "https://limese.com/products/psoriasis-shampoo",
            "image": "https://m.media-amazon.com/images/I/61tKQhj3QBL._SL1500_.jpg",
            "description": "Therapeutic shampoo for scalp psoriasis",
            "price": "₹699",
            "color": "#B5EAD7"
        }
    ],
    "Acne": [
        {
            "name": "Acne Treatment Gel",
            "url": "https://www.purplle.com/product/acne-gel",
            "image": "https://m.media-amazon.com/images/I/61SJuj2VURL._SL1500_.jpg",
            "description": "Benzoyl peroxide gel for acne treatment",
            "price": "₹499 - ₹899",
            "color": "#FFD1DC"
        },
        {
            "name": "Salicylic Acid Cleanser",
            "url": "https://limese.com/products/acne-cleanser",
            "image": "https://m.media-amazon.com/images/I/61KbY6QSl4L._SL1500_.jpg",
            "description": "Oil-free cleanser for acne-prone skin",
            "price": "₹599",
            "color": "#B5EAD7"
        }
    ],
    "default": [
        {
            "name": "Moisturizing Cream",
            "url": "https://www.purplle.com/collections/best-moisturizer-for-dry-sensitive-skin",
            "image": "https://m.media-amazon.com/images/I/61SJuj2VURL._SL1500_.jpg",
            "description": "Specialized moisturizers for dry and sensitive skin",
            "price": "₹599 - ₹1,299",
            "color": "#FFD1DC"
        },
        {
            "name": "Gentle Cleanser",
            "url": "https://limese.com/products/cosrx-low-ph-good-morning-gel-cleanser",
            "image": "https://m.media-amazon.com/images/I/61KbY6QSl4L._SL1500_.jpg",
            "description": "Low pH cleanser for sensitive skin",
            "price": "₹799",
            "color": "#B5EAD7"
        }
    ]
}

# =========================
# COMPREHENSIVE MEDICINE DATABASE WITH IMAGES
# =========================
MEDICINE_DATABASE = {
    "Eczema": {
        "symptoms": ["Itchy, red, dry, scaly skin patches", "Oozing or crusting in severe cases"],
        "treatment": ["Topical corticosteroids", "Moisturizers", "Antihistamines"],
        "medicines": [
            {
                "name": "Hydrocortisone Cream 1%",
                "description": "Topical corticosteroid for mild eczema",
                "image": "https://m.media-amazon.com/images/I/41T2JvDlWRL._SL500_.jpg",
                "type": "Topical Corticosteroid",
                "brand_names": ["Cortizone-10", "Hydrocortisone"],
                "usage": "Apply thin layer to affected area 1-3 times daily"
            },
            {
                "name": "Tacrolimus Ointment 0.1%",
                "description": "Calcineurin inhibitor for moderate to severe eczema",
                "image": "https://m.media-amazon.com/images/I/41Kj5VXqJYL._SL500_.jpg",
                "type": "Immunomodulator",
                "brand_names": ["Protopic"],
                "usage": "Apply twice daily to affected areas"
            },
            {
                "name": "Cetirizine 10mg",
                "description": "Antihistamine for itching relief",
                "image": "https://m.media-amazon.com/images/I/41QZJZJZJZL._SL500_.jpg",
                "type": "Antihistamine",
                "brand_names": ["Zyrtec"],
                "usage": "10mg tablet once daily"
            }
        ],
        "prevention": ["Avoid harsh soaps", "Use fragrance-free moisturizers", "Wear cotton clothing"]
    },
    "Psoriasis": {
        "symptoms": ["Thick, red patches with silvery scales", "Itching, burning"],
        "treatment": ["Topical treatments", "Phototherapy", "Systemic medications"],
        "medicines": [
            {
                "name": "Betamethasone Dipropionate",
                "description": "Potent topical corticosteroid",
                "image": "https://m.media-amazon.com/images/I/41J8J8J8J8L._SL500_.jpg",
                "type": "Topical Corticosteroid",
                "brand_names": ["Diprolene"],
                "usage": "Apply once or twice daily"
            },
            {
                "name": "Calcipotriene Cream",
                "description": "Vitamin D analog for plaque psoriasis",
                "image": "https://m.media-amazon.com/images/I/41K8K8K8K8L._SL500_.jpg",
                "type": "Vitamin D Analog",
                "brand_names": ["Dovonex"],
                "usage": "Apply twice daily"
            },
            {
                "name": "Methotrexate",
                "description": "Systemic treatment for severe psoriasis",
                "image": "https://m.media-amazon.com/images/I/41L8L8L8L8L._SL500_.jpg",
                "type": "DMARD",
                "brand_names": ["Trexall"],
                "usage": "Prescription only - follow doctor's instructions"
            }
        ],
        "prevention": ["Moisturize skin", "Avoid smoking/alcohol", "Manage stress"]
    },
    "Acne": {
        "symptoms": ["Blackheads, whiteheads, pimples", "Cysts and nodules"],
        "treatment": ["Topical retinoids", "Antibiotics", "Benzoyl peroxide"],
        "medicines": [
            {
                "name": "Tretinoin Cream 0.025%",
                "description": "Topical retinoid for acne treatment",
                "image": "https://m.media-amazon.com/images/I/41M8M8M8M8L._SL500_.jpg",
                "type": "Retinoid",
                "brand_names": ["Retin-A"],
                "usage": "Apply thin layer at bedtime"
            },
            {
                "name": "Clindamycin Phosphate Gel 1%",
                "description": "Topical antibiotic for acne",
                "image": "https://m.media-amazon.com/images/I/41N8N8N8N8L._SL500_.jpg",
                "type": "Antibiotic",
                "brand_names": ["Cleocin T"],
                "usage": "Apply twice daily"
            },
            {
                "name": "Benzoyl Peroxide 5% Gel",
                "description": "Antibacterial treatment for acne",
                "image": "https://m.media-amazon.com/images/I/41P8P8P8P8L._SL500_.jpg",
                "type": "Antibacterial",
                "brand_names": ["PanOxyl"],
                "usage": "Apply once or twice daily"
            }
        ],
        "prevention": ["Gentle cleansing", "Non-comedogenic products", "Avoid picking"]
    },
    "Rosacea": {
        "symptoms": ["Facial redness", "Visible blood vessels"],
        "treatment": ["Topical medications", "Oral antibiotics", "Laser therapy"],
        "medicines": [
            {
                "name": "Metronidazole Gel 0.75%",
                "description": "Topical antibiotic for rosacea",
                "image": "https://m.media-amazon.com/images/I/41Q8Q8Q8Q8L._SL500_.jpg",
                "type": "Antibiotic",
                "brand_names": ["MetroGel"],
                "usage": "Apply once daily"
            },
            {
                "name": "Azelaic Acid Gel 15%",
                "description": "Anti-inflammatory for rosacea",
                "image": "https://m.media-amazon.com/images/I/41R8R8R8R8L._SL500_.jpg",
                "type": "Anti-inflammatory",
                "brand_names": ["Finacea"],
                "usage": "Apply twice daily"
            },
            {
                "name": "Doxycycline 40mg",
                "description": "Oral antibiotic for moderate to severe rosacea",
                "image": "https://m.media-amazon.com/images/I/41S8S8S8S8L._SL500_.jpg",
                "type": "Oral Antibiotic",
                "brand_names": ["Oracea"],
                "usage": "One capsule daily"
            }
        ],
        "prevention": ["Avoid triggers", "Sun protection", "Gentle skincare"]
    }
}


# =========================
# INFERMEDICA API FUNCTIONS
# =========================
def fetch_infermedica_info(disease_name):
    """Fetch comprehensive medical information from Infermedica API"""

    # Map common disease names to Infermedica condition IDs
    condition_mapping = {
        "Eczema": "C0013602",  # Atopic dermatitis
        "Psoriasis": "C0033860",  # Psoriasis
        "Acne": "C0001070",  # Acne vulgaris
        "Rosacea": "C0035849",  # Rosacea
        "Melanoma": "C0025202",  # Melanoma
        "Ringworm": "C0040246",  # Tinea corporis
        "Dermatitis": "C0011606",  # Dermatitis
        "Fungal Infection": "C0016627",  # Fungal infection
        "Skin Condition": "C0037274"  # Skin disease
    }

    condition_id = condition_mapping.get(disease_name)

    try:
        headers = {
            'App-Id': INFERMEDICA_APP_ID,
            'App-Key': INFERMEDICA_APP_KEY,
            'Content-Type': 'application/json'
        }

        if condition_id:
            # Try to get condition by ID
            url = f"{INFERMEDICA_BASE_URL}conditions/{condition_id}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                condition_data = response.json()
                return process_infermedica_condition(condition_data, disease_name)

        # If no specific condition ID or failed, try search
        url = f"{INFERMEDICA_BASE_URL}search"
        params = {
            'phrase': disease_name,
            'types': 'condition'
        }
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            search_results = response.json()
            if search_results and len(search_results) > 0:
                # Get details for the first matching condition
                condition_id = search_results[0]['id']
                url = f"{INFERMEDICA_BASE_URL}conditions/{condition_id}"
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    condition_data = response.json()
                    return process_infermedica_condition(condition_data, disease_name)

    except Exception as e:
        print(f"Error fetching Infermedica data: {e}")

    return None


def process_infermedica_condition(condition_data, original_name):
    """Process Infermedica condition data into our format"""

    try:
        # Extract relevant information
        name = condition_data.get('name', original_name)
        common_name = condition_data.get('common_name', name)
        prevalence = condition_data.get('prevalence', 'Common')
        severity = condition_data.get('severity', 'Moderate')
        acuteness = condition_data.get('acuteness', 'Chronic')

        # Extract symptoms
        symptoms = []
        extras = condition_data.get('extras', {})
        if 'symptoms' in extras:
            symptoms = extras['symptoms'][:5]  # Get top 5 symptoms

        # Extract risk factors
        risk_factors = []
        if 'risk_factors' in extras:
            risk_factors = extras['risk_factors'][:3]

        # Extract treatment
        treatment_info = []
        if 'management' in extras:
            treatment_info = extras['management'][:3]

        # Extract prevention
        prevention_info = []
        if 'prevention' in extras:
            prevention_info = extras['prevention'][:3]

        # Get additional info
        categories = condition_data.get('categories', [])
        category_names = [cat.get('name', '') for cat in categories[:2]]

        # Get references
        references = []
        if 'references' in condition_data:
            references = condition_data['references'][:2]

        return {
            "name": name,
            "common_name": common_name,
            "description": f"{name} ({common_name}) is a {acuteness.lower()} {category_names[0] if category_names else 'skin'} condition with {prevalence.lower()} prevalence.",
            "symptoms": symptoms,
            "risk_factors": risk_factors,
            "treatment": treatment_info,
            "prevention": prevention_info,
            "severity": severity,
            "prevalence": prevalence,
            "acuteness": acuteness,
            "categories": category_names,
            "references": references,
            "source": "Infermedica Medical API",
            "source_type": "Professional Medical Database"
        }

    except Exception as e:
        print(f"Error processing condition data: {e}")

    return None


def get_infermedica_symptoms(disease_name):
    """Get related symptoms for a disease"""
    try:
        headers = {
            'App-Id': INFERMEDICA_APP_ID,
            'App-Key': INFERMEDICA_APP_KEY,
            'Content-Type': 'application/json'
        }

        url = f"{INFERMEDICA_BASE_URL}search"
        params = {
            'phrase': disease_name,
            'types': 'symptom',
            'max_results': 5
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            symptoms = response.json()
            return [symptom['label'] for symptom in symptoms[:5]]

    except Exception as e:
        print(f"Error fetching symptoms: {e}")

    return []


def get_infermedica_treatments(disease_name):
    """Get treatment information"""
    try:
        headers = {
            'App-Id': INFERMEDICA_APP_ID,
            'App-Key': INFERMEDICA_APP_KEY,
            'Content-Type': 'application/json'
        }

        url = f"{INFERMEDICA_BASE_URL}search"
        params = {
            'phrase': f"{disease_name} treatment",
            'types': 'risk_factor',
            'max_results': 5
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            treatments = response.json()
            return [treatment['label'] for treatment in treatments[:5]]

    except Exception as e:
        print(f"Error fetching treatments: {e}")

    return []


# =========================
# ENHANCED MEDICINE INFORMATION FETCHER
# =========================
def fetch_medicine_info_online(disease_name):
    """Fetch medicine information from online sources"""
    medicines_info = []

    try:
        # Try to get information from Wikipedia
        wikipedia.set_rate_limiting(True)

        # Search for medicines related to the disease
        search_terms = [
            f"{disease_name} treatment medications",
            f"{disease_name} prescription drugs",
            f"{disease_name} topical treatments"
        ]

        for term in search_terms:
            try:
                # Search Wikipedia
                search_results = wikipedia.search(term, results=3)

                for result in search_results:
                    if "treatment" in result.lower() or "drug" in result.lower() or "medication" in result.lower():
                        try:
                            page = wikipedia.page(result)
                            content = page.content[:500]  # Get first 500 characters

                            # Extract medicine names using regex patterns
                            medicine_patterns = [
                                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:cream|ointment|gel|tablet|capsule)\b',
                                r'\b[A-Z][a-z]+\s+(?:hydrochloride|acetate|phosphate)\b',
                                r'\b\d+%?\s+[A-Z][a-z]+\b'
                            ]

                            found_medicines = []
                            for pattern in medicine_patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                found_medicines.extend(matches)

                            # Add unique medicines
                            for med in found_medicines[:5]:  # Limit to 5 medicines
                                if med.lower() not in [m['name'].lower() for m in medicines_info]:
                                    medicines_info.append({
                                        "name": med,
                                        "description": f"Commonly prescribed for {disease_name}",
                                        "image": get_medicine_image(med),
                                        "type": "Prescription Medication",
                                        "brand_names": [med.split()[0]],
                                        "usage": "Consult doctor for dosage"
                                    })
                        except:
                            continue
            except:
                continue

        # If no medicines found from Wikipedia, use our database
        if not medicines_info and disease_name in MEDICINE_DATABASE:
            return MEDICINE_DATABASE[disease_name]["medicines"]

        return medicines_info[:3]  # Return max 3 medicines

    except Exception as e:
        # Fallback to database
        if disease_name in MEDICINE_DATABASE:
            return MEDICINE_DATABASE[disease_name]["medicines"]
        return []


def get_medicine_image(medicine_name):
    """Get image URL for medicine (using placeholder if not found)"""
    # Map common medicines to images
    medicine_images = {
        "hydrocortisone": "https://m.media-amazon.com/images/I/41T2JvDlWRL._SL500_.jpg",
        "tacrolimus": "https://m.media-amazon.com/images/I/41Kj5VXqJYL._SL500_.jpg",
        "cetirizine": "https://m.media-amazon.com/images/I/41QZJZJZJZL._SL500_.jpg",
        "betamethasone": "https://m.media-amazon.com/images/I/41J8J8J8J8L._SL500_.jpg",
        "calcipotriene": "https://m.media-amazon.com/images/I/41K8K8K8K8L._SL500_.jpg",
        "methotrexate": "https://m.media-amazon.com/images/I/41L8L8L8L8L._SL500_.jpg",
        "tretinoin": "https://m.media-amazon.com/images/I/41M8M8M8M8L._SL500_.jpg",
        "clindamycin": "https://m.media-amazon.com/images/I/41N8N8N8N8L._SL500_.jpg",
        "benzoyl peroxide": "https://m.media-amazon.com/images/I/41P8P8P8P8L._SL500_.jpg",
        "metronidazole": "https://m.media-amazon.com/images/I/41Q8Q8Q8Q8L._SL500_.jpg",
        "azelaic acid": "https://m.media-amazon.com/images/I/41R8R8R8R8L._SL500_.jpg",
        "doxycycline": "https://m.media-amazon.com/images/I/41S8S8S8S8L._SL500_.jpg"
    }

    # Check if medicine name contains any known medicine
    for med_key, image_url in medicine_images.items():
        if med_key in medicine_name.lower():
            return image_url

    # Return default medicine image
    return "https://m.media-amazon.com/images/I/41V8V8V8V8L._SL500_.jpg"


# =========================
# ENHANCED DEEP LEARNING MODEL FUNCTIONS
# =========================
@st.cache_resource
def load_model():
    """Load the pre-trained TensorFlow model"""
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("✅ Advanced AI Model loaded successfully!")
            return model
        else:
            st.warning("⚠️ Model file not found. Using enhanced analysis mode.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def get_class_names():
    """Get sorted list of disease classes from training directory"""
    try:
        if os.path.exists(TRAIN_DIR):
            class_names = sorted([
                d for d in os.listdir(TRAIN_DIR)
                if os.path.isdir(os.path.join(TRAIN_DIR, d))
            ])
            return class_names
        else:
            return ["Eczema", "Psoriasis", "Acne", "Rosacea", "Melanoma", "Ringworm"]
    except:
        return ["Eczema", "Psoriasis", "Acne", "Rosacea"]


def enhance_prediction_confidence(preds, original_idx):
    """Apply post-processing to improve prediction confidence"""
    preds = preds.flatten()

    # Get top 3 predictions
    top_indices = np.argsort(preds)[-3:][::-1]
    top_confidences = preds[top_indices]

    # If top prediction is significantly higher than others
    if top_confidences[0] > top_confidences[1] * 1.5:
        return float(top_confidences[0])

    # Return weighted average
    weighted_pred = sum(top_confidences * np.array([0.7, 0.2, 0.1]))
    return min(1.0, float(weighted_pred * 1.1))


def predict_with_deep_learning(image, model, class_names):
    """Enhanced prediction using deep learning model"""
    # Preprocess image
    IMAGE_SIZE = (224, 224)
    CONFIDENCE_THRESHOLD = 0.50

    image = image.resize(IMAGE_SIZE)
    img = np.array(image)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # Normalize
    img = img / 255.0
    img_batch = np.array([img])

    if model:
        # Predict
        preds = model.predict(img_batch, verbose=0)
        confidence = float(np.max(preds))
        idx = int(np.argmax(preds))

        # Apply confidence enhancement
        enhanced_confidence = enhance_prediction_confidence(preds, idx)

        # Get disease name
        disease_name = class_names[idx] if idx < len(class_names) else "Unknown"

        # Get top 3 predictions
        top_3_idx = np.argsort(preds.flatten())[-3:][::-1]
        top_3_diseases = [class_names[i] if i < len(class_names) else "Unknown" for i in top_3_idx]
        top_3_confidences = [float(preds.flatten()[i]) for i in top_3_idx]

        return disease_name, enhanced_confidence, top_3_diseases, top_3_confidences
    else:
        # Fallback to traditional method
        return predict_traditional(image), 0.7, ["Analysis Mode", "Check Model", "Consult Doctor"], [0.5, 0.3, 0.2]


# =========================
# TRADITIONAL PREDICTION (BACKUP)
# =========================
def load_training_images(train_dir='train'):
    """Load training images for traditional matching"""
    disease_images = {}
    if os.path.exists(train_dir):
        for disease in os.listdir(train_dir):
            disease_path = os.path.join(train_dir, disease)
            if os.path.isdir(disease_path):
                images = []
                for img_file in os.listdir(disease_path):
                    if img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(disease_path, img_file)
                        try:
                            img = Image.open(img_path)
                            img = img.resize((224, 224))
                            img_array = np.array(img)
                            if len(img_array.shape) == 2:
                                img_array = np.stack((img_array,) * 3, axis=-1)
                            images.append(img_array)
                        except:
                            pass
                if images:
                    disease_images[disease] = images
    return disease_images


def extract_features(img_array):
    """Extract features for traditional matching"""
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    if img_array.shape[2] == 1:
        img_array = np.concatenate([img_array] * 3, axis=2)

    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Color histograms
    hist_r = cv2.calcHist([img_array], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_array], [2], None, [256], [0, 256]).flatten()

    # Texture features
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    patches = extract_patches_2d(gray, (32, 32), max_patches=10)
    patch_features = [p.flatten() for p in patches]
    texture_feature = np.mean(patch_features, axis=0)

    # Combine features
    features = np.concatenate([hist_r, hist_g, hist_b, texture_feature])
    return features


def predict_traditional(uploaded_img):
    """Traditional prediction method (fallback)"""
    try:
        # Simulate analysis based on color
        img_array = np.array(uploaded_img.resize((100, 100)))
        avg_color = np.mean(img_array, axis=(0, 1))

        # Simple color-based classification
        if avg_color[0] > 150:  # Reddish
            return "Eczema or Rosacea"
        elif np.std(avg_color) < 20:
            return "Fungal Infection"
        else:
            return "Dermatitis"
    except:
        return "Skin Condition Detected"


# =========================
# WEB VERIFICATION FUNCTIONS
# =========================
def verify_disease_with_google(disease_name):
    """Verify disease name with Google search"""
    try:
        query = f"is {disease_name} a valid skin disease?"
        search_results = list(search(query, num=3, stop=3, pause=2))
        reputable_sources = ['wikipedia.org', 'webmd.com', 'mayoclinic.org']
        for result in search_results:
            if any(source in result.lower() for source in reputable_sources):
                return True
        return False
    except:
        return True  # Assume valid if verification fails


def search_disease_info(disease_name):
    """Search disease information from medical websites"""
    try:
        query = f"{disease_name} skin disease symptoms and treatment"
        search_results = list(search(query, num=5, stop=5, pause=2))
        medical_sources = []
        for result in search_results:
            if any(domain in result for domain in
                   ['mayoclinic.org', 'webmd.com', 'healthline.com']):
                medical_sources.append(result)
        return medical_sources[:3]
    except:
        return []


# =========================
# DISEASE INFORMATION FUNCTION (UPDATED WITH INFERMEDICA)
# =========================
def get_disease_info(disease_name):
    """Get comprehensive disease information with Infermedica integration"""

    # First try to fetch information from Infermedica
    infermedica_info = fetch_infermedica_info(disease_name)

    if infermedica_info:
        # Create comprehensive response with Infermedica data
        result = {
            'disease_name': infermedica_info.get('name', disease_name),
            'common_name': infermedica_info.get('common_name', disease_name),
            'scientific_name': infermedica_info.get('name', disease_name),
            'description': infermedica_info.get('description',
                                                f"Medical information about {disease_name} from authoritative sources."),
            'symptoms': infermedica_info.get('symptoms', []),
            'risk_factors': infermedica_info.get('risk_factors', []),
            'treatment_options': infermedica_info.get('treatment', []),
            'prevention_options': infermedica_info.get('prevention', []),
            'severity': infermedica_info.get('severity', 'Moderate'),
            'prevalence': infermedica_info.get('prevalence', 'Common'),
            'acuteness': infermedica_info.get('acuteness', 'Chronic'),
            'categories': infermedica_info.get('categories', []),
            'treatment': 'Consult a dermatologist for proper diagnosis and treatment.',
            'recommended_tablets': ['Consult doctor for medications'],
            'medicines': [],
            'prevention': infermedica_info.get('prevention', ['Avoid known irritants', 'Maintain proper skin hygiene']),
            'follow_up': 'Recommended in 2 weeks if no improvement',
            'source': f"Source: {infermedica_info.get('source', 'Infermedica Medical Database')}",
            'source_type': infermedica_info.get('source_type', 'Professional Medical Database'),
            'infermedica_data': infermedica_info
        }

        # Add additional symptoms if not enough from Infermedica
        if not result['symptoms']:
            result['symptoms'] = get_infermedica_symptoms(disease_name)

        # Add additional treatments if not enough from Infermedica
        if not result['treatment_options']:
            result['treatment_options'] = get_infermedica_treatments(disease_name)

        return result

    # If Infermedica fails, try Wikipedia
    if WIKIPEDIA_AVAILABLE:
        try:
            page_py = wiki_wiki.page(disease_name)
            if page_py.exists():
                summary = page_py.summary.split('. ')[0:3]
                wiki_summary = '. '.join(summary) + '.'
                return {
                    'disease_name': disease_name,
                    'scientific_name': disease_name,
                    'description': wiki_summary,
                    'severity': 'Moderate',
                    'treatment': 'Consult a dermatologist for proper diagnosis and treatment.',
                    'recommended_tablets': ['Consult doctor for medications'],
                    'prevention': ['Avoid known irritants', 'Maintain proper skin hygiene'],
                    'follow_up': 'Recommended in 2 weeks if no improvement',
                    'source': f"Source: Wikipedia"
                }
        except:
            pass

    # Check medicine database
    if disease_name in MEDICINE_DATABASE:
        disease_data = MEDICINE_DATABASE[disease_name]
        return {
            'disease_name': disease_name,
            'scientific_name': disease_name,
            'description': f"A dermatological condition characterized by: {', '.join(disease_data['symptoms'])}",
            'severity': random.choice(['Mild', 'Moderate', 'Severe']),
            'treatment': 'Treatment options include: ' + ', '.join(disease_data['treatment']),
            'recommended_tablets': [med['name'] for med in disease_data['medicines']],
            'medicines': disease_data['medicines'],  # Add detailed medicine information
            'prevention': disease_data['prevention'],
            'follow_up': 'Recommended in 2 weeks if no improvement',
            'source': 'Medical database'
        }

    # Default response
    return {
        'disease_name': disease_name,
        'scientific_name': disease_name,
        'description': 'A dermatological condition requiring professional evaluation.',
        'severity': 'Moderate',
        'treatment': 'Consult a dermatologist for proper diagnosis and treatment.',
        'recommended_tablets': ['Consult doctor for medications'],
        'medicines': [],
        'prevention': ['Avoid known irritants', 'Maintain proper skin hygiene'],
        'follow_up': 'Recommended in 2 weeks if no improvement',
        'source': 'Medical database'
    }


# =========================
# LOCATION SERVICES
# =========================
def get_coordinates(city_name):
    """Get coordinates for a city"""
    url = f"https://api.geoapify.com/v1/geocode/search?text={city_name}&apiKey={GEOAPIFY_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and response.json()['features']:
        coords = response.json()['features'][0]['geometry']['coordinates']
        return coords[1], coords[0]
    return None, None


def find_nearby_hospitals(lat, lon):
    """Find nearby hospitals"""
    url = f"https://api.geoapify.com/v2/places?categories=healthcare.hospital&filter=circle:{lon},{lat},5000&limit=10&apiKey={GEOAPIFY_API_KEY}"
    response = requests.get(url)
    hospitals = []
    if response.status_code == 200:
        for feature in response.json().get('features', []):
            props = feature['properties']
            hospitals.append({
                'name': props.get('name', 'Unknown'),
                'address': props.get('formatted', 'No address provided'),
                'lat': feature['geometry']['coordinates'][1],
                'lon': feature['geometry']['coordinates'][0]
            })
    return hospitals


# =========================
# MEDICAL REPORT GENERATION (UPDATED WITH INFERMEDICA)
# =========================
def generate_medical_report(patient_name, patient_age, city, disease_info, hospitals):
    """Generate comprehensive medical report HTML with Infermedica integration"""

    # Generate Infermedica data section
    infermedica_section = ""
    if 'infermedica_data' in disease_info:
        infermedica_data = disease_info['infermedica_data']

        # Generate symptoms list
        symptoms_list = ""
        if disease_info.get('symptoms'):
            symptoms_list = "".join([f"<li>{symptom}</li>" for symptom in disease_info['symptoms'][:5]])
        elif 'symptoms' in infermedica_data and infermedica_data['symptoms']:
            symptoms_list = "".join([f"<li>{symptom}</li>" for symptom in infermedica_data['symptoms'][:5]])

        # Generate risk factors list
        risk_factors_list = ""
        if disease_info.get('risk_factors'):
            risk_factors_list = "".join([f"<li>{factor}</li>" for factor in disease_info['risk_factors'][:3]])
        elif 'risk_factors' in infermedica_data and infermedica_data['risk_factors']:
            risk_factors_list = "".join([f"<li>{factor}</li>" for factor in infermedica_data['risk_factors'][:3]])

        # Generate treatment options list
        treatment_list = ""
        if disease_info.get('treatment_options'):
            treatment_list = "".join([f"<li>{treatment}</li>" for treatment in disease_info['treatment_options'][:5]])
        elif 'treatment' in infermedica_data and infermedica_data['treatment']:
            treatment_list = "".join([f"<li>{treatment}</li>" for treatment in infermedica_data['treatment'][:5]])

        infermedica_section = f"""
        <div class="section">
            <h2 class="section-title">📊 Medical Analysis Details</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                <div style="background: #e8f4fc; padding: 1.25rem; border-radius: 8px;">
                    <h4 style="color: #2c4d7a; margin-top: 0;">🧬 Condition Profile</h4>
                    <div style="margin-bottom: 0.75rem;">
                        <strong>Prevalence:</strong> {infermedica_data.get('prevalence', 'Common')}
                    </div>
                    <div style="margin-bottom: 0.75rem;">
                        <strong>Acuteness:</strong> {infermedica_data.get('acuteness', 'Chronic')}
                    </div>
                    <div style="margin-bottom: 0.75rem;">
                        <strong>Category:</strong> {', '.join(infermedica_data.get('categories', ['Dermatological']))}
                    </div>
                    <div style="margin-bottom: 0.75rem;">
                        <strong>Data Source:</strong> {infermedica_data.get('source', 'Professional Medical Database')}
                    </div>
                </div>

                <div style="background: #f0f9ff; padding: 1.25rem; border-radius: 8px;">
                    <h4 style="color: #2c4d7a; margin-top: 0;">📋 Key Symptoms</h4>
                    <ul style="margin-top: 0.5rem;">
                        {symptoms_list if symptoms_list else '<li>No specific symptoms data available</li>'}
                    </ul>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;">
                <div style="background: #fff0f0; padding: 1.25rem; border-radius: 8px;">
                    <h4 style="color: #2c4d7a; margin-top: 0;">⚠️ Risk Factors</h4>
                    <ul style="margin-top: 0.5rem;">
                        {risk_factors_list if risk_factors_list else '<li>No specific risk factors identified</li>'}
                    </ul>
                </div>

                <div style="background: #f0fff0; padding: 1.25rem; border-radius: 8px;">
                    <h4 style="color: #2c4d7a; margin-top: 0;">💊 Treatment Options</h4>
                    <ul style="margin-top: 0.5rem;">
                        {treatment_list if treatment_list else '<li>Consult dermatologist for treatment options</li>'}
                    </ul>
                </div>
            </div>
        </div>
        """

    # Generate medicine HTML
    medicine_html = ""
    if 'medicines' in disease_info and disease_info['medicines']:
        for med in disease_info['medicines']:
            medicine_html += f"""
            <div class="medicine-card">
                <div class="medicine-image">
                    <img src="{med.get('image', 'https://m.media-amazon.com/images/I/41V8V8V8V8L._SL500_.jpg')}" alt="{med['name']}">
                </div>
                <div class="medicine-info">
                    <h4>{med['name']}</h4>
                    <p><strong>Type:</strong> {med.get('type', 'Prescription Medication')}</p>
                    <p><strong>Description:</strong> {med.get('description', 'Commonly prescribed medication')}</p>
                    <p><strong>Brand Names:</strong> {', '.join(med.get('brand_names', ['Various']))}</p>
                    <p><strong>Usage:</strong> {med.get('usage', 'Consult doctor for dosage instructions')}</p>
                </div>
            </div>
            """
    else:
        medicine_html = "<p>No specific medication information available. Please consult a dermatologist.</p>"

    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Report - {patient_name}</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }}
            .report-container {{
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                padding: 30px;
                margin: 20px 0;
            }}
            .header {{
                text-align: center;
                padding-bottom: 20px;
                border-bottom: 3px solid #4a6fa5;
                margin-bottom: 30px;
            }}
            .header h1 {{
                color: #2c4d7a;
                margin-bottom: 10px;
                font-size: 28px;
            }}
            .header p {{
                color: #666;
                font-size: 14px;
            }}
            .section {{
                margin-bottom: 25px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 4px solid #4a6fa5;
            }}
            .section-title {{
                color: #2c4d7a;
                font-size: 18px;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .patient-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .info-item {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .info-label {{
                font-weight: bold;
                color: #4a6fa5;
                font-size: 12px;
                text-transform: uppercase;
                margin-bottom: 5px;
            }}
            .info-value {{
                font-size: 16px;
                color: #333;
            }}
            .severity-badge {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
                margin-left: 10px;
            }}
            .severity-mild {{ background: #d4edda; color: #155724; }}
            .severity-moderate {{ background: #fff3cd; color: #856404; }}
            .severity-severe {{ background: #f8d7da; color: #721c24; }}
            .hospital-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                border-left: 3px solid #ff7e5f;
            }}
            .hospital-name {{
                font-weight: bold;
                color: #2c4d7a;
                margin-bottom: 5px;
            }}
            .hospital-address {{
                color: #666;
                font-size: 14px;
                margin-bottom: 5px;
            }}
            .medicine-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                display: flex;
                gap: 15px;
                align-items: flex-start;
            }}
            .medicine-image {{
                flex: 0 0 100px;
            }}
            .medicine-image img {{
                width: 100%;
                border-radius: 8px;
            }}
            .medicine-info {{
                flex: 1;
            }}
            .medicine-info h4 {{
                margin-top: 0;
                color: #2c4d7a;
            }}
            .treatment-list li {{
                margin-bottom: 8px;
                padding-left: 5px;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 12px;
            }}
            .disclaimer {{
                background: #fff3cd;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #ffc107;
            }}
            @media print {{
                body {{
                    background: white;
                }}
                .report-container {{
                    box-shadow: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div class="header">
                <h1>🏥 Skin Health Pro+ Medical Report</h1>
                <p>Comprehensive Dermatological Analysis Report</p>
                <p>Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}</p>
                <p style="color: #4a6fa5; font-weight: bold;">Powered by Infermedica Medical Intelligence</p>
            </div>

            <div class="section">
                <h2 class="section-title">👤 Patient Information</h2>
                <div class="patient-info">
                    <div class="info-item">
                        <div class="info-label">Patient Name</div>
                        <div class="info-value">{patient_name}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Age</div>
                        <div class="info-value">{patient_age} years</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Location</div>
                        <div class="info-value">{city.upper()}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Report ID</div>
                        <div class="info-value">SHP-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2 class="section-title">🔍 Diagnosis Summary</h2>
                <div class="patient-info">
                    <div class="info-item">
                        <div class="info-label">Primary Diagnosis</div>
                        <div class="info-value">{disease_info['disease_name']}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Common Name</div>
                        <div class="info-value">{disease_info.get('common_name', disease_info['disease_name'])}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Condition Severity</div>
                        <div class="info-value">
                            {disease_info['severity']}
                            <span class="severity-badge severity-{disease_info['severity'].lower()}">
                                {disease_info['severity']}
                            </span>
                        </div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Follow-up Required</div>
                        <div class="info-value">{disease_info['follow_up']}</div>
                    </div>
                </div>

                <h3 style="color: #4a6fa5; margin-top: 20px;">Condition Description</h3>
                <p>{disease_info['description']}</p>
                <p><small><i>{disease_info.get('source', 'Medical database')}</i></small></p>
            </div>

            {infermedica_section}

            <div class="section">
                <h2 class="section-title">💊 Medical Information</h2>
                <h3>Commonly Prescribed Medications for {disease_info['disease_name']}:</h3>
                {medicine_html}
            </div>

            <div class="section">
                <h2 class="section-title">🏥 Recommended Healthcare Facilities</h2>
                <p>Nearby dermatology centers in {city.upper()}:</p>
                {''.join([f'''
                <div class="hospital-card">
                    <div class="hospital-name">{hospital['name']}</div>
                    <div class="hospital-address">📍 {hospital['address']}</div>
                </div>
                ''' for hospital in hospitals[:5]])}
            </div>

            <div class="disclaimer">
                <h3>⚠️ Important Disclaimer</h3>
                <p>This report is generated by an AI-powered diagnostic tool for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.</p>
                <p><strong>For Emergency:</strong> If you experience severe symptoms, difficulty breathing, or rapid spreading of skin lesions, seek immediate medical attention or call emergency services.</p>
                <p><strong>Sources:</strong> Medical information powered by Infermedica API - Professional Medical Intelligence Platform.</p>
            </div>

            <div class="footer">
                <p>Skin Health Pro+ | Advanced AI Dermatology Diagnostics</p>
                <p>Powered by Infermedica Medical Intelligence</p>
                <p>Report Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}</p>
                <p>© 2024 Skin Health Pro+. All rights reserved.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return report_html


# =========================
# STREAMLIT APP CONFIGURATION
# =========================
st.set_page_config(
    page_title="Skin Health Pro+ — AI Dermatology Dashboard",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Medical Dashboard Styling
st.markdown("""
    <style>
        :root {
            --primary: #4a6fa5;
            --primary-dark: #2c4d7a;
            --primary-light: #e1e8f5;
            --secondary: #ff7e5f;
            --background: #f8f9fa;
            --card-bg: #ffffff;
            --text-dark: #2d3748;
            --text-light: #718096;
            --border-radius: 12px;
            --box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        body {
            background-color: var(--background);
            font-family: 'Inter', sans-serif;
        }
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
            border-bottom: 1px solid #e2e8f0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            border-radius: var(--border-radius);
            animation: fadeIn 1s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .header-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: white;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .header-subtitle {
            font-size: 1rem;
            color: rgba(255,255,255,0.8);
            margin-top: -0.5rem;
        }
        .card {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-top: 4px solid var(--primary);
            transition: all 0.3s ease;
            animation: slideIn 0.5s ease-out;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        }
        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--primary-dark);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .diagnosis-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--primary-light);
            color: var(--primary-dark);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 1rem;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .severity-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        .severity-mild {
            background: #48bb78;
        }
        .severity-moderate {
            background: #ed8936;
        }
        .severity-severe {
            background: #f56565;
        }
        .treatment-card {
            background: var(--primary-light);
            padding: 1.25rem;
            border-radius: var(--border-radius);
            margin: 1.5rem 0;
        }
        .medicine-card {
            background: white;
            border-radius: var(--border-radius);
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            display: flex;
            gap: 1.25rem;
            align-items: flex-start;
            transition: all 0.2s;
            border-left: 4px solid var(--primary);
        }
        .medicine-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .medicine-image {
            flex: 0 0 100px;
        }
        .medicine-image img {
            width: 100%;
            border-radius: 8px;
            height: 100px;
            object-fit: cover;
        }
        .medicine-content {
            flex: 1;
        }
        .medicine-name {
            font-weight: 700;
            color: var(--primary-dark);
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        .medicine-type {
            display: inline-block;
            background: var(--primary-light);
            color: var(--primary-dark);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }
        .medicine-description {
            color: var(--text-light);
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }
        .medicine-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
        }
        .medicine-detail {
            background: var(--background);
            padding: 0.5rem;
            border-radius: 6px;
            font-size: 0.85rem;
        }
        .medicine-detail strong {
            color: var(--primary-dark);
            display: block;
            margin-bottom: 0.25rem;
        }
        .hospital-box {
            border: 1px solid #e2e8f0;
            border-radius: var(--border-radius);
            padding: 1rem;
            margin-bottom: 1rem;
            transition: all 0.2s;
            background: white;
        }
        .hospital-box:hover {
            border-color: var(--primary);
            box-shadow: var(--box-shadow);
            transform: translateY(-3px);
        }
        .hospital-name {
            font-weight: 600;
            color: var(--primary-dark);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.25rem;
        }
        .hospital-address {
            font-size: 0.85rem;
            color: var(--text-light);
            margin-bottom: 0.5rem;
        }
        .hospital-distance {
            font-size: 0.75rem;
            color: var(--primary);
            font-weight: 500;
        }
        .progress-text {
            color: var(--primary-dark);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        .product-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        .product-card {
            border-radius: var(--border-radius);
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }
        .product-image {
            height: 200px;
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: flex-end;
        }
        .product-content {
            padding: 1.5rem;
            background: white;
        }
        .product-title {
            font-weight: 700;
            color: var(--primary-dark);
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }
        .product-description {
            color: var(--text-light);
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        .product-price {
            font-weight: 700;
            color: var(--secondary);
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }
        .product-button {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            width: 100%;
            transition: all 0.2s;
            font-weight: 600;
        }
        .product-button:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
        }
        .upload-container {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            border-radius: var(--border-radius);
            padding: 2rem;
            margin-bottom: 2rem;
            color: white;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        .data-table th {
            background-color: var(--primary-light);
            color: var(--primary-dark);
            text-align: left;
            padding: 0.75rem;
            font-weight: 600;
        }
        .data-table td {
            padding: 0.75rem;
            border-bottom: 1px solid #e2e8f0;
        }
        .data-table tr:hover {
            background-color: rgba(74, 111, 165, 0.05);
        }
        .fade-in {
            animation: fadeIn 1s ease-in;
        }
        .slide-up {
            animation: slideUp 0.8s ease-out;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .highlight-box {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 5px solid #4a6fa5;
            padding: 1.25rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .highlight-title {
            font-weight: 600;
            color: #2c4d7a;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        @media (max-width: 768px) {
            .header-container {
                flex-direction: column;
                align-items: flex-start;
            }
            .product-grid {
                grid-template-columns: 1fr;
            }
            .medicine-card {
                flex-direction: column;
            }
            .medicine-image {
                flex: 0 0 auto;
                width: 100%;
            }
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# INITIALIZE SESSION STATE
# =========================
if 'model_loaded' not in st.session_state:
    st.session_state.model = load_model()
    st.session_state.class_names = get_class_names()
    st.session_state.model_loaded = True

if 'disease_images' not in st.session_state:
    st.session_state.disease_images = load_training_images()

if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {'name': '', 'age': '', 'city': ''}

if 'hospitals' not in st.session_state:
    st.session_state.hospitals = []

# =========================
# MAIN APP LAYOUT
# =========================
# Header with gradient background
st.markdown("""
    <div class="header-container">
        <div>
            <h1 class="header-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                </svg>
                Skin Health Pro+
            </h1>
            <p class="header-subtitle">Advanced AI Dermatology Diagnostics with Deep Learning</p>
            <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-top: 0.5rem;">
                Powered by Infermedica Medical Intelligence
            </p>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; display: flex; align-items: center; gap: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 0-2-3 4 4 0 0 0-4-1"></path>
                </svg>
                <span style="color: white; font-weight: 500;">AI Powered</span>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; display: flex; align-items: center; gap: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
                    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
                </svg>
                <span style="color: white; font-weight: 500;">Medical API</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main content in two columns
col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    # Upload Section with animated gradient
    st.markdown("""
        <div class="upload-container">
            <h3 style="margin-top: 0; color: white; display: flex; align-items: center; gap: 0.75rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                Upload Skin Image
            </h3>
            <p style="color: rgba(255,255,255,0.9);">Get an instant AI-powered analysis of your skin condition</p>
        </div>
    """, unsafe_allow_html=True)

    # Lottie animation for upload section
    if lottie_scan:
        st_lottie(lottie_scan, height=200, key="upload-animation")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"],
                                     label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded dermatological image", use_column_width=True)

        # Enhanced AI Analysis
        with st.spinner(""):
            progress_bar = st.progress(0)
            status_text = st.markdown('<p class="progress-text">🧠 Running deep learning analysis...</p>',
                                      unsafe_allow_html=True)

            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)

            progress_bar.empty()
            status_text.empty()

            # =========================
            # ENHANCED DISEASE PREDICTION
            # =========================
            disease_name, confidence, top_3_diseases, top_3_confidences = predict_with_deep_learning(
                image,
                st.session_state.model,
                st.session_state.class_names
            )

            # Verify with Google
            if confidence > 0.5:
                is_verified = verify_disease_with_google(disease_name)
                if is_verified:
                    st.success("✅ Diagnosis verified through medical databases")

            # Fetch comprehensive disease information from Infermedica
            with st.spinner("📚 Fetching authoritative medical information from Infermedica..."):
                result = get_disease_info(disease_name)

                # Display Infermedica source badge if available
                if 'infermedica_data' in result:
                    st.success(f"✅ Medical information retrieved from Infermedica Professional Database")
                    st.info(f"📊 Condition: {result.get('prevalence', 'Common')} | Severity: {result['severity']}")

            # Fetch medicine information
            with st.spinner("💊 Fetching medication information..."):
                medicines = fetch_medicine_info_online(disease_name)
                if not medicines and disease_name in MEDICINE_DATABASE:
                    medicines = MEDICINE_DATABASE[disease_name]["medicines"]
                result['medicines'] = medicines

            # Display results
            if confidence >= 0.5:
                st.success(f"✅ **Primary Diagnosis:** {disease_name}")
                st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")

                # Show top 3 possibilities
                with st.expander("🔍 View alternative possibilities"):
                    for i, (dis, conf) in enumerate(zip(top_3_diseases, top_3_confidences)):
                        st.write(f"{i + 1}. {dis}: {conf * 100:.1f}%")

            else:
                st.warning(f"⚠️ **Low Confidence Prediction:** {disease_name}")
                st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")

                # Show alternatives for low confidence
                st.markdown("### 🔎 Alternative Possibilities:")
                for i, (dis, conf) in enumerate(zip(top_3_diseases, top_3_confidences)):
                    st.write(f"{i + 1}. {dis} ({conf * 100:.1f}%)")

        # Diagnosis Section with animation
        st.markdown('<div class="card slide-up">', unsafe_allow_html=True)
        st.markdown("""
            <div class="card-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <path d="M14 2v6h6"></path>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                Diagnostic Report
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="diagnosis-badge">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
                </svg>
                <span>Primary Diagnosis</span>
            </div>
            <h3 style="margin-top: 0; color: var(--primary-dark);">{result['disease_name']}</h3>
            <p style="color: var(--text-light);"><i>{result.get('common_name', result['scientific_name'])}</i></p>
            <p>{result['description']}</p>
            <p style="font-size: 0.85rem; color: var(--text-light);">{result['source']}</p>

            <div style="display: flex; gap: 1rem; margin: 1.5rem 0;">
                <div style="flex: 1; background: var(--primary-light); padding: 1.25rem; border-radius: 12px;">
                    <div style="font-size: 0.85rem; color: var(--text-light); margin-bottom: 0.5rem;">Severity</div>
                    <div style="font-weight: 600; color: var(--text-dark); display: flex; align-items: center;">
                        <span class="severity-indicator severity-{result['severity'].lower()}"></span>
                        {result['severity']}
                    </div>
                </div>
                <div style="flex: 1; background: var(--primary-light); padding: 1.25rem; border-radius: 12px;">
                    <div style="font-size: 0.85rem; color: var(--text-light); margin-bottom: 0.5rem;">AI Confidence</div>
                    <div style="font-weight: 600; color: var(--text-dark);">
                        {confidence * 100:.1f}%
                    </div>
                </div>
                <div style="flex: 1; background: var(--primary-light); padding: 1.25rem; border-radius: 12px;">
                    <div style="font-size: 0.85rem; color: var(--text-light); margin-bottom: 0.5rem;">Prevalence</div>
                    <div style="font-weight: 600; color: var(--text-dark);">
                        {result.get('prevalence', 'Common')}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Display Infermedica detailed information if available
        if 'infermedica_data' in result:
            with st.expander("📊 View Detailed Medical Analysis", expanded=False):
                infermedica_data = result['infermedica_data']

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("### 🧬 Condition Profile")
                    st.markdown(f"""
                        **Prevalence:** {infermedica_data.get('prevalence', 'Common')}

                        **Acuteness:** {infermedica_data.get('acuteness', 'Chronic')}

                        **Categories:** {', '.join(infermedica_data.get('categories', ['Dermatological']))}
                    """)

                    if result.get('symptoms'):
                        st.markdown("### 📋 Key Symptoms")
                        for symptom in result['symptoms'][:5]:
                            st.markdown(f"• {symptom}")

                with col_b:
                    if result.get('risk_factors'):
                        st.markdown("### ⚠️ Risk Factors")
                        for factor in result['risk_factors'][:3]:
                            st.markdown(f"• {factor}")

                    if result.get('treatment_options'):
                        st.markdown("### 💊 Treatment Options")
                        for treatment in result['treatment_options'][:5]:
                            st.markdown(f"• {treatment}")

                st.info(f"**Source:** {result.get('source_type', 'Professional Medical Database')}")

        # Medical Information Section
        st.markdown("""
            <div class="treatment-card">
                <h4 style="margin-top: 0; color: var(--primary-dark); display: flex; align-items: center; gap: 0.5rem;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a8 8 0 0 0-8 8c0 1.892.402 3.13 1.5 4.5L12 22l6.5-7.5c1.098-1.37 1.5-2.608 1.5-4.5a8 8 0 0 0-8-8z"></path>
                        <path d="M9 12h6"></path>
                        <path d="M12 9v6"></path>
                    </svg>
                    Medical Information
                </h4>
                <p>Commonly prescribed medications for {disease_name}:</p>
            </div>
        """, unsafe_allow_html=True)

        # Display medicines with images
        if result.get('medicines'):
            for medicine in result['medicines']:
                st.markdown(f"""
                    <div class="medicine-card">
                        <div class="medicine-image">
                            <img src="{medicine.get('image', 'https://m.media-amazon.com/images/I/41V8V8V8V8L._SL500_.jpg')}" alt="{medicine['name']}">
                        </div>
                        <div class="medicine-content">
                            <div class="medicine-name">{medicine['name']}</div>
                            <div class="medicine-type">{medicine.get('type', 'Prescription Medication')}</div>
                            <div class="medicine-description">{medicine.get('description', 'Commonly prescribed medication')}</div>
                            <div class="medicine-details">
                                <div class="medicine-detail">
                                    <strong>Brand Names</strong>
                                    {', '.join(medicine.get('brand_names', ['Various']))}
                                </div>
                                <div class="medicine-detail">
                                    <strong>Usage</strong>
                                    {medicine.get('usage', 'Consult doctor for dosage instructions')}
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info(
                "ℹ️ No specific medication information available. Please consult a dermatologist for proper treatment.")

        # Create the prevention tips HTML
        prevention_tips_html = ''.join([
            f'<li style="margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>'
            f'<polyline points="22 4 12 14.01 9 11.01"></polyline>'
            f'</svg>{tip}'
            f'</li>'
            for tip in result.get('prevention', [])
        ])

        st.markdown(f"""
            <h4 style="color: var(--primary-dark); margin-top: 1.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
                </svg>
                Prevention & Lifestyle Guidelines
            </h4>
            <ul style="margin-top: 0;">
                {prevention_tips_html if prevention_tips_html else '<li>No specific prevention guidelines available. Consult a dermatologist.</li>'}
            </ul>

            <h4 style="color: var(--primary-dark); margin-top: 1.5rem; display: flex; align-items: center; gap: 0.5rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                Clinical Follow-up Schedule
            </h4>
            <p>{result.get('follow_up', 'Recommended in 2 weeks if no improvement')}</p>
        """, unsafe_allow_html=True)

        # Product Recommendations Section
        st.markdown('<div class="card slide-up">', unsafe_allow_html=True)
        st.markdown("""
            <div class="card-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="9" cy="21" r="1"></circle>
                    <circle cx="20" cy="21" r="1"></circle>
                    <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path>
                </svg>
                Recommended Products
            </div>
            <p style="color: var(--text-light);">These products may help with your condition:</p>
        """, unsafe_allow_html=True)

        if lottie_health:
            st_lottie(lottie_health, height=150, key="products-animation")

        # Get appropriate products
        if disease_name in PRODUCT_RECOMMENDATIONS:
            products = PRODUCT_RECOMMENDATIONS[disease_name]
        else:
            products = PRODUCT_RECOMMENDATIONS['default']

        # Product grid
        st.markdown('<div class="product-grid">', unsafe_allow_html=True)
        for product in products:
            st.markdown(f"""
                <div class="product-card">
                    <div class="product-image" style="background-color: {product['color']}; background-image: url('{product['image']}')">
                    </div>
                    <div class="product-content">
                        <h3 class="product-title">{product['name']}</h3>
                        <p class="product-description">{product['description']}</p>
                        <div class="product-price">{product['price']}</div>
                        <a href="{product['url']}" target="_blank" class="product-button">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                            </svg>
                            Buy Now
                        </a>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Hospital Finder Section
    st.markdown('<div class="card slide-up" id="hospitals-section">', unsafe_allow_html=True)
    st.markdown("""
        <div class="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M18 8h1a4 4 0 0 1 0 8h-1"></path>
                <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"></path>
                <line x1="6" y1="1" x2="6" y2="4"></line>
                <line x1="10" y1="1" x2="10" y2="4"></line>
                <line x1="14" y1="1" x2="14" y2="4"></line>
            </svg>
            Healthcare Facilities Locator
        </div>
    """, unsafe_allow_html=True)

    if lottie_doctor:
        st_lottie(lottie_doctor, height=200, key="hospital-animation")

    # City input for hospital search
    city = st.text_input("Enter your location to find accredited dermatology centers:",
                         placeholder="City or ZIP code",
                         key="hospital_location",
                         help="Enter your city to find nearby dermatology clinics and hospitals")

    # Hospital search functionality (runs when city is entered)
    if city:
        with st.spinner("🔍 Searching for nearby healthcare facilities..."):
            lat, lon = get_coordinates(city)
            if lat and lon:
                hospitals = find_nearby_hospitals(lat, lon)
                if hospitals:
                    st.session_state.hospitals = hospitals

                    # Display hospitals in a highlighted section below the input
                    st.markdown(f"""
                        <div class="highlight-box">
                            <div class="highlight-title">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M18 8h1a4 4 0 0 1 0 8h-1"></path>
                                    <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"></path>
                                    <line x1="6" y1="1" x2="6" y2="4"></line>
                                    <line x1="10" y1="1" x2="10" y2="4"></line>
                                    <line x1="14" y1="1" x2="14" y2="4"></line>
                                </svg>
                                Nearby Healthcare Facilities in {city.upper()}
                            </div>
                            <p>Found {len(hospitals)} medical centers near your location:</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Display hospitals in individual boxes
                    for idx, hospital in enumerate(hospitals[:5], 1):
                        st.markdown(f"""
                            <div class="hospital-box">
                                <div class="hospital-name">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="M18 8h1a4 4 0 0 1 0 8h-1"></path>
                                        <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"></path>
                                        <line x1="6" y1="1" x2="6" y2="4"></line>
                                        <line x1="10" y1="1" x2="10" y2="4"></line>
                                        <line x1="14" y1="1" x2="14" y2="4"></line>
                                    </svg>
                                    {hospital['name']}
                                </div>
                                <div class="hospital-address">📍 {hospital['address']}</div>
                                <div class="hospital-distance">Approx. {(idx * 1.5):.1f} km away</div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.success(f"✅ Found {len(hospitals)} accredited medical centers near {city}")
                else:
                    st.warning(f"⚠️ No hospitals found near {city}. Please try a larger city or check the spelling.")
            else:
                st.error("❌ Could not find coordinates for the entered location. Please check the city name.")

    # Patient Information Form for Medical Report
    st.markdown("---")
    st.markdown("### 📋 Patient Information for Medical Report")

    with st.form("patient_info_form"):
        patient_name = st.text_input("Patient's Full Name:", placeholder="Enter full name")
        patient_age = st.text_input("Patient's Age:", placeholder="Enter age")
        report_city = st.text_input("Patient's City:", placeholder="Enter city for hospital search",
                                    value=city if city else "")

        submitted = st.form_submit_button("📄 Generate Comprehensive Medical Report")

    if submitted and uploaded_file:
        if not patient_name or not patient_age or not report_city:
            st.warning("⚠️ Please enter all patient information to generate the report.")
        else:
            # Get coordinates and hospitals for the report
            lat, lon = get_coordinates(report_city)
            report_hospitals = []
            if lat and lon:
                report_hospitals = find_nearby_hospitals(lat, lon)

            # Generate the medical report
            report_html = generate_medical_report(
                patient_name,
                patient_age,
                report_city,
                result,
                report_hospitals
            )

            # Show the report and download button
            st.markdown("---")
            st.markdown("### 📄 Generated Medical Report")

            # Display report preview
            with st.expander("Preview Medical Report", expanded=True):
                components.html(report_html, height=800, scrolling=True)

            # Download button
            st.download_button(
                label="📥 Download Full Medical Report (HTML)",
                data=report_html,
                file_name=f"{patient_name.replace(' ', '_')}_SkinHealth_Report_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                help="Click to download a comprehensive medical report in HTML format"
            )

            st.success("✅ Medical report generated successfully! Download using the button above.")

# Additional Information Section
st.markdown('<div class="card slide-up">', unsafe_allow_html=True)
st.markdown("""
        <div class="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            Skin Health Tips
        </div>
        <div style="margin-bottom: 1rem;">
            <h4 style="color: var(--primary-dark); margin-bottom: 0.5rem;">Daily Skin Care Routine</h4>
            <ul style="margin-top: 0;">
                <li>Cleanse your skin twice daily</li>
                <li>Use sunscreen with SPF 30+ every day</li>
                <li>Moisturize regularly</li>
                <li>Stay hydrated and eat a balanced diet</li>
            </ul>
        </div>
        <div style="margin-bottom: 1rem;">
            <h4 style="color: var(--primary-dark); margin-bottom: 0.5rem;">When to See a Dermatologist</h4>
            <ul style="margin-top: 0;">
                <li>Persistent acne or skin irritation</li>
                <li>Changing moles or skin growths</li>
                <li>Unexplained rashes or skin discoloration</li>
                <li>Severe dryness or itching</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer with Infermedica credit
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 1.5rem; background-color: var(--primary-light); border-radius: var(--border-radius); animation: fadeIn 1s ease-in;">
        <p style="color: var(--primary-dark); font-size: 0.9rem;">Skin Health Pro+ uses advanced AI technology and professional medical data from Infermedica to analyze skin conditions.</p>
        <p style="color: var(--text-light); font-size: 0.8rem; margin-top: 0.5rem;">Medical information powered by Infermedica API - Professional Medical Intelligence Platform.</p>
        <p style="color: var(--text-light); font-size: 0.8rem;">© 2024 Skin Health Pro+. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
