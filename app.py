import os
import torch
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import re

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "outputs/fast_model"
DATA_ROOT = "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TSV_FILES = [
    os.path.join(DATA_ROOT, "test", "test.tsv"),
    os.path.join(DATA_ROOT, "train", "train.tsv"),
    os.path.join(DATA_ROOT, "dev", "dev.tsv")
]

print("="*50)
print(f"🚀 INITIALIZING VALIDATION APP (Clean Mode) ON {DEVICE}")
print("="*50)

# 1. LOAD MODEL
try:
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")
    exit(1)

# 2. BUILD MASTER DB (ROBUST LOADING)
ground_truth_db = {}

print("📂 Building Master Ground Truth Database...")
for tsv_path in TSV_FILES:
    if os.path.exists(tsv_path):
        try:
            # quoting=3 (QUOTE_NONE) prevents parsing errors if text contains quotes
            df = pd.read_csv(tsv_path, sep='\t', quoting=3, on_bad_lines='skip')
            
            # Clean columns
            df.columns = [c.strip() for c in df.columns]
            
            count = 0
            for _, row in df.iterrows():
                raw_filename = str(row['image']).strip()
                raw_text = str(row['text']).strip()
                
                # Store strictly by filename (e.g., "img_123.jpg")
                clean_key = os.path.basename(raw_filename)
                ground_truth_db[clean_key] = raw_text
                count += 1
                
            print(f"   ✓ Loaded {count} entries from {os.path.basename(tsv_path)}")
        except Exception as e:
            print(f"   ⚠️ Error reading {tsv_path}: {e}")

print(f"✅ Database Ready: {len(ground_truth_db)} images indexed.")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tigrinya OCR Validator</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background-color: #f4f7f6; 
            margin: 0; padding: 40px; 
            display: flex; flex-direction: column; align-items: center; 
            min-height: 100vh;
        }
        
        .main-card { 
            background: white; padding: 40px; border-radius: 15px; 
            box-shadow: 0 10px 25px rgba(0,0,0,0.1); 
            width: 100%; max-width: 800px; text-align: center; margin-bottom: 30px;
        }

        h1 { color: #2c3e50; margin-bottom: 10px; }
        p { color: #7f8c8d; margin-bottom: 30px; }

        .btn { 
            border: 2px solid #3498db; color: white; background-color: #3498db; 
            padding: 12px 30px; border-radius: 8px; font-size: 16px; font-weight: bold; 
            cursor: pointer; transition: 0.3s; display: inline-block; position: relative;
        }
        .btn:hover { background-color: #2980b9; border-color: #2980b9; }
        .file-input { position: absolute; left: 0; top: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
        
        .btn-clear { background-color: transparent; color: #e74c3c; border-color: #e74c3c; margin-top: 10px; }
        .btn-clear:hover { background-color: #e74c3c; color: white; }

        #results-container { width: 100%; max-width: 800px; display: flex; flex-direction: column; gap: 15px; }

        /* Result Card Styles */
        .result-item { 
            background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
            display: flex; align-items: center; padding: 15px; 
            animation: fadeIn 0.5s ease-out; border-left: 6px solid #bdc3c7; 
        }

        /* Status Colors */
        .result-item.processing { border-left-color: #f39c12; } 
        .result-item.correct { border-left-color: #2ecc71; background-color: #f0fdf4; } 
        .result-item.incorrect { border-left-color: #e74c3c; background-color: #fef2f2; } 
        .result-item.unknown { border-left-color: #3498db; background-color: #f0f7ff; } 

        .thumb-box { 
            flex: 0 0 120px; height: 90px; display: flex; align-items: center; 
            justify-content: center; background: #fff; border-radius: 8px; 
            overflow: hidden; margin-right: 20px; border: 1px solid #eee;
        }
        .thumb-box img { max-width: 100%; max-height: 100%; object-fit: contain; }

        .text-box { flex: 1; text-align: left; }
        
        .filename-label { font-size: 11px; color: #95a5a6; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
        
        .ocr-text { 
            font-size: 22px; color: #2c3e50; font-family: 'Nyala', 'Abyssinica SIL', serif; font-weight: bold; 
        }
        
        /* Badges */
        .badge { font-size: 12px; padding: 3px 8px; border-radius: 4px; vertical-align: middle; margin-right: 8px; }
        
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>

    <div class="main-card">
        <h1>🇪🇷 Tigrinya OCR Validator</h1>
        <p>Upload images from your dataset to validate.</p>
        
        <div style="position: relative; display: inline-block;">
            <button class="btn">📂 Upload Images</button>
            <input type="file" id="fileInput" class="file-input" accept="image/*" multiple>
        </div>
        <br>
        <button class="btn btn-clear" onclick="clearResults()" id="clearBtn" style="display:none;">🗑️ Clear List</button>
    </div>

    <div id="results-container"></div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const container = document.getElementById('results-container');
        const clearBtn = document.getElementById('clearBtn');

        fileInput.onchange = async (evt) => {
            const files = fileInput.files;
            if (files.length === 0) return;
            clearBtn.style.display = "inline-block";

            for (let i = 0; i < files.length; i++) {
                createResultCard(files[i]);
                await processFile(files[i]);
            }
            fileInput.value = "";
        };

        function createResultCard(file) {
            const id = file.name.replace(/[^a-zA-Z0-9]/g, '');
            const div = document.createElement('div');
            div.className = 'result-item processing';
            div.id = 'card-' + id;
            const imgUrl = URL.createObjectURL(file);

            div.innerHTML = `
                <div class="thumb-box"><img src="${imgUrl}"></div>
                <div class="text-box">
                    <div class="filename-label">${file.name}</div>
                    <div class="ocr-text" id="text-${id}">⏳ Processing...</div>
                </div>
            `;
            container.prepend(div);
        }

        async function processFile(file) {
            const id = file.name.replace(/[^a-zA-Z0-9]/g, '');
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                
                const card = document.getElementById('card-' + id);
                const textBox = document.getElementById('text-' + id);
                
                textBox.innerText = data.pred;

                // --- LOGIC: Color Coding ---
                if (data.status === "correct") {
                    card.className = 'result-item correct';
                    textBox.innerHTML = "✅ " + data.pred;
                } 
                else if (data.status === "incorrect") {
                    card.className = 'result-item incorrect';
                    textBox.innerHTML = "❌ " + data.pred;
                    // No "Expected" text here, just Red Card.
                } 
                else {
                    // Unknown (Not in dataset)
                    card.className = 'result-item unknown';
                    textBox.innerHTML = "⚠️ " + data.pred;
                }

            } catch (error) {
                document.getElementById('text-' + id).innerText = "Error";
            }
        }

        function clearResults() { container.innerHTML = ""; clearBtn.style.display = "none"; }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file'})
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file'})
    
    try:
        # 1. Prediction
        image = Image.open(file.stream).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, max_length=128, num_beams=5, 
                repetition_penalty=1.2, no_repeat_ngram_size=3, early_stopping=True
            )
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # 2. Validation Logic
        # Robust filename matching
        clean_filename = re.sub(r'^\d+_', '', file.filename) # Remove "1_" prefixes
        clean_filename = os.path.basename(clean_filename)    # Remove folder paths
        
        ground_truth = ground_truth_db.get(clean_filename)
        
        status = "unknown"
        if ground_truth:
            # Normalize comparison (Trim spaces)
            if pred_text.strip() == ground_truth.strip():
                status = "correct"
            else:
                status = "incorrect"
        
        return jsonify({
            'pred': pred_text,
            'status': status
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)