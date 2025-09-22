# deepfake_detector_app.py
# Enhanced Streamlit Deepfake Detector with Professional UI

import streamlit as st
from PIL import Image
import io, os, tempfile, time, math, re
import numpy as np
import cv2

# Transformers & audio
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import librosa
import soundfile as sf
import fitz  # PyMuPDF

# Add these imports for better error handling
import traceback
from typing import List, Dict, Any, Tuple, Optional, Union

# Set page configuration
st.set_page_config(
    page_title="DeepGuard AI - Content Authenticity Verifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Default model IDs (Hugging Face)
# -------------------------
# Audio model
DEFAULT_AUDIO_MODEL = "microsoft/wavlm-base"

# Image deepfake detectors
DEFAULT_IMAGE_MODELS = [
    "prithivMLmods/deepfake-detector-model-v1",
    "microsoft/resnet-50"
]

# Text LM for perplexity
DEFAULT_TEXT_MODEL = "gpt2"

# -------------------------
# Helper utilities
# -------------------------
@st.cache_resource(show_spinner=False)
def get_image_pipelines(model_ids):
    """Return list of HF image-classification pipelines (local)."""
    pipes = {}
    for mid in model_ids:
        try:
            with st.spinner(f"🔄 Loading image model: {mid}"):
                pipes[mid] = pipeline("image-classification", model=mid, device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            st.warning(f"⚠️ Failed to load model {mid}: {str(e)}")
            # try CPU fallback
            try:
                pipes[mid] = pipeline("image-classification", model=mid, device=-1)
            except Exception as e2:
                st.error(f"❌ Completely failed to load model {mid}: {str(e2)}")
    return pipes

@st.cache_resource(show_spinner=False)
def get_audio_pipeline(model_id):
    try:
        with st.spinner(f"🔄 Loading audio model: {model_id}"):
            return pipeline("audio-classification", model=model_id, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.warning(f"⚠️ Failed to load audio model {model_id}: {str(e)}")
        try:
            return pipeline("audio-classification", model=model_id, device=-1)
        except Exception as e2:
            st.error(f"❌ Completely failed to load audio model {model_id}: {str(e2)}")
            return None

@st.cache_resource(show_spinner=False)
def get_gpt2_model_and_tokenizer(model_name):
    try:
        with st.spinner(f"🔄 Loading text model: {model_name}"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(model_name)
            if torch.cuda.is_available():
                model = model.to("cuda")
            return tokenizer, model
    except Exception as e:
        st.error(f"❌ Failed to load text model {model_name}: {str(e)}")
        return None, None

# Basic stylometry heuristics
def stylometry_score(text: str):
    if not text or len(text.strip()) == 0:
        return 0.0
        
    tokens = re.findall(r"\w+", text)
    n = max(1, len(tokens))
    uniq_ratio = len(set(t.lower() for t in tokens))/n
    # bigram repetition
    bigrams = list(zip(tokens, tokens[1:]))
    rep_ratio = 0.0
    if bigrams:
        rep_ratio = (len(bigrams) - len(set((a.lower(), b.lower()) for a,b in bigrams))) / len(bigrams)
    # avg sentence length
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_sent_len = np.mean([len(re.findall(r"\w+", s)) for s in sents]) if sents else 0
    score = 50.0
    if uniq_ratio < 0.45: score += 18
    if rep_ratio > 0.05: score += 10
    if avg_sent_len < 6: score += 5
    return max(0.0, min(100.0, score))

def compute_perplexity(text: str, tokenizer, model):
    if not text or not tokenizer or not model:
        return float('inf')
        
    # Truncate to model max length to avoid OOM
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        model.to("cuda")
        
    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()
        ppl = math.exp(loss) if loss < 100 else float("inf")
        return ppl
    except Exception as e:
        st.error(f"❌ Error computing perplexity: {str(e)}")
        return float('inf')

# Map HF classifier output to an AI-likelihood score 0..100
def parse_label_score(hf_pred_list):
    """
    HF image/audio classifier returns list of dicts [{label,score},...]
    We interpret labels containing 'fake', 'deepfake', 'manipulated' etc as evidence of AI.
    Return score in 0..100 where higher means more likely AI/fake.
    """
    if not hf_pred_list:
        return None
    top = hf_pred_list[0]
    label = str(top.get("label","")).lower()
    score = float(top.get("score", 0.0))
    fake_words = ["fake","deepfake","spoof","manipulated","synth","synthetic","generated"]
    real_words = ["real","genuine","bonafide","human","clean","authentic","original","photo"]
    if any(w in label for w in fake_words):
        return score * 100.0
    if any(w in label for w in real_words):
        return (1.0 - score) * 100.0
    # otherwise, heuristically treat the model's confidence as indicative of AI if label isn't explicitly 'real'
    return score * 100.0

# Video frame extraction (sample up to n frames evenly)
def extract_frames_from_video_bytes(video_bytes, max_frames=8):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp.write(video_bytes); tmp.flush(); tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # Try to get frames by reading until failure
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(pil)
            total = len(frames)
            if total == 0:
                return [Image.new('RGB', (224, 224), color='red')]  # Return a dummy image
            
        if total > 0:
            indices = np.linspace(0, max(0,total-1), min(max_frames, total)).astype(int)
            idx_set = set(indices.tolist())
            frames = []
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if i in idx_set:
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(pil)
                i += 1
            cap.release()
            return frames if frames else [Image.new('RGB', (224, 224), color='red')]
        else:
            return [Image.new('RGB', (224, 224), color='red')]
    except Exception as e:
        st.error(f"❌ Error extracting video frames: {str(e)}")
        return [Image.new('RGB', (224, 224), color='red')]
    finally:
        try: 
            os.unlink(tmp.name)
        except: 
            pass

# PDF analysis (structural + visual)
def analyze_pdf_bytes(pdf_bytes):
    ai_score = 50.0
    details = {}
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        details["page_count"] = doc.page_count
        fonts = set()
        for i in range(min(5, doc.page_count)):
            page = doc.load_page(i)
            try:
                for f in page.get_fonts(full=True):
                    fonts.add(str(f))
            except Exception:
                pass
        details["unique_fonts"] = len(fonts)
        if len(fonts) > 12:
            ai_score += 10
        eof_count = len(re.findall(br"%%EOF", pdf_bytes))
        details["eof_markers"] = int(eof_count)
        if eof_count > 1:
            ai_score += 8
        visual = []
        for i in range(min(2, doc.page_count)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5,1.5))
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 180)
            edge_density = float(np.mean(edges>0))
            visual.append({"page": i+1, "edge_density": edge_density})
            if edge_density < 0.005:
                ai_score += 5
        details["visual_findings"] = visual
    except Exception as e:
        details["error"] = str(e)
    ai_score = max(0.0, min(100.0, ai_score))
    verdict = "Likely Manipulated" if ai_score >= 60 else "Likely Authentic"
    return {"score": ai_score, "verdict": verdict, "details": details}

# -------------------------
# UI & main orchestration
# -------------------------

# Header section
st.markdown('<h1 class="main-header">🛡️ DeepGuard AI</h1>', unsafe_allow_html=True)
st.markdown("### Your Trusted Content Authenticity Verifier")
st.markdown("""
<div class="info-box">
    <p>Welcome to DeepGuard AI! This tool helps you verify the authenticity of images, videos, audio, text, and documents.</p>
    <p>Simply upload a file or paste text below to check if it might be AI-generated or manipulated.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/dusk/128/000000/security-checked.png", width=80)
    st.markdown("## Settings ⚙️")
    
    with st.expander("Advanced Settings", expanded=False):
        image_model_ids = st.text_area("Image Model IDs", value=", ".join(DEFAULT_IMAGE_MODELS), 
                                      help="Technical identifiers for image analysis models")
        audio_model_id = st.text_input("Audio Model ID", value=DEFAULT_AUDIO_MODEL)
        text_model_name = st.text_input("Text Model", value=DEFAULT_TEXT_MODEL)
        max_frames = st.slider("Video Frames to Analyze", 1, 16, 8)
    
    threshold = st.slider("Confidence Threshold", 0, 100, 60, 
                         help="Higher values mean stricter detection of AI-generated content")
    show_details = st.checkbox("Show Technical Details", value=False)
    
    st.markdown("---")
    st.markdown("### About ℹ️")
    st.markdown("""
    DeepGuard AI uses advanced machine learning models to detect AI-generated or manipulated content.
    All processing happens locally on your device for maximum privacy.
    """)
    
    st.markdown("---")
    st.markdown("#### Supported Formats:")
    st.markdown("- **Images**: JPG, PNG, WebP")
    st.markdown("- **Audio**: WAV, MP3, M4A, FLAC")
    st.markdown("- **Video**: MP4, MOV, AVI, MKV")
    st.markdown("- **Documents**: PDF")
    st.markdown("- **Text**: Direct input")

# lazy model loading helpers
image_model_list = [m.strip() for m in image_model_ids.split(",") if m.strip()]
img_pipes = get_image_pipelines(image_model_list)
audio_pipe = get_audio_pipeline(audio_model_id)
tokenizer, gpt2_model = get_gpt2_model_and_tokenizer(text_model_name)

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Upload Content 📁")
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    upload = st.file_uploader("Choose a file to analyze", 
                             type=["png", "jpg", "jpeg", "mp4", "mov", "avi", "mkv", "wav", "mp3", "m4a", "flac", "pdf"],
                             help="Select an image, video, audio file, or document",
                             label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Or Enter Text 📝")
    text_input = st.text_area("Paste text content to analyze", height=120,
                             placeholder="Enter text here to check if it might be AI-generated...")
    
    analyze_btn = st.button("Analyze Content", type="primary", use_container_width=True)

with col2:
    results_placeholder = st.empty()
    
    if not analyze_btn and not upload and not text_input:
        with results_placeholder.container():
            st.markdown("""
            <div class="info-box">
                <h3>👆 Get Started</h3>
                <p>1. Upload a file or paste text in the left panel</p>
                <p>2. Click the 'Analyze Content' button</p>
                <p>3. View your authenticity results here</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### How It Works 🔍")
            st.markdown("""
            - **Images & Videos**: Analyzes visual patterns to detect AI generation
            - **Audio**: Checks for synthetic voice patterns and audio artifacts
            - **Text**: Examines writing style and complexity for AI patterns
            - **Documents**: Looks for structural and formatting inconsistencies
            """)
            
            st.markdown("### Why It Matters 🌐")
            st.markdown("""
            In today's digital world, it's increasingly important to verify the authenticity of content.
            DeepGuard AI helps you identify potentially AI-generated or manipulated media.
            """)

# Analysis functions with improved audio processing
def run_image_pipeline(img_bytes):
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        per_model_scores = []
        details = {}
        for mid, pipe in img_pipes.items():
            try:
                preds = pipe(pil)
                sc = parse_label_score(preds)
                if sc is None:
                    # fallback: use top score as proxy
                    sc = float(preds[0].get("score", 0.0)) * 100.0
                per_model_scores.append(sc)
                details[mid] = preds
            except Exception as e:
                details[mid] = {"error": str(e)}
        if per_model_scores:
            score = float(np.mean(per_model_scores))
        else:
            score = None
        return score, per_model_scores, details, pil
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, [], {"error": str(e)}, None

def run_audio_pipeline(audio_bytes):
    if audio_pipe is None:
        return None, {"error": "Audio model not loaded"}
    
    # Create a temporary file with proper extension
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        # Try to load with librosa which handles many formats
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            # Write as WAV file
            sf.write(tmp.name, y, sr, subtype='PCM_16')
        except Exception as e:
            return None, {"error": f"Failed to process audio: {str(e)}"}
        
        # Now use the pipeline on the WAV file
        preds = audio_pipe(tmp.name)
        sc = parse_label_score(preds)
        if sc is None:
            sc = float(preds[0].get("score",0.0))*100.0
        details = preds
        return sc, details
    except Exception as e:
        return None, {"error": f"Failed to process audio: {str(e)}"}
    finally:
        try: 
            os.unlink(tmp.name)
        except: 
            pass

def run_video_pipeline(video_bytes, max_frames=8):
    frames = extract_frames_from_video_bytes(video_bytes, max_frames=max_frames)
    if not frames:
        return None, {"error":"no frames extracted"}
    frame_scores = []
    frame_details = []
    for pil in frames:
        # run image ensemble on the frame
        per_model_scores = []
        details = {}
        for mid, pipe in img_pipes.items():
            try:
                preds = pipe(pil)
                sc = parse_label_score(preds)
                if sc is None:
                    sc = float(preds[0].get("score",0.0))*100.0
                per_model_scores.append(sc)
                details[mid] = preds
            except Exception as e:
                details[mid] = {"error": str(e)}
        frame_scores.append(float(np.mean(per_model_scores)) if per_model_scores else None)
        frame_details.append(details)
    valid = [s for s in frame_scores if s is not None]
    mean_score = float(np.mean(valid)) if valid else None
    top_k = max(1, int(math.ceil(0.2 * len(valid)))) if valid else 1
    topk_mean = float(np.mean(sorted(valid, reverse=True)[:top_k])) if valid else None
    final = max(mean_score or 0.0, topk_mean or 0.0)
    return final, {"frame_scores": frame_scores, "frame_details": frame_details, "mean": mean_score, "topk_mean": topk_mean}

def run_text_pipeline(text):
    if not text or not tokenizer or not gpt2_model:
        return 0.0, {"error": "Text model not loaded"}
        
    sty = stylometry_score(text)
    ppl = compute_perplexity(text, tokenizer, gpt2_model)
    # map ppl to a 0-100 contribution (calibration needed)
    if ppl == float("inf"):
        ppl_score = 0.0
    elif ppl < 20:
        ppl_score = 40.0
    elif ppl < 50:
        ppl_score = 20.0
    else:
        ppl_score = 0.0
    score = min(100.0, 0.6*sty + 0.4*ppl_score)
    verdict = "Likely AI" if score >= threshold else "Likely Human"
    return score, {"stylometry": sty, "perplexity": ppl, "verdict": verdict}

# Main analysis logic
if analyze_btn and (upload or text_input):
    start = time.time()
    
    with results_placeholder.container():
        st.markdown("## Analysis Results 📊")
        
        if text_input and not upload:
            with st.spinner("🔍 Analyzing text content..."):
                score, details = run_text_pipeline(text_input)
                
            if "error" in details:
                st.markdown(f'<div class="error-box">{details["error"]}</div>', unsafe_allow_html=True)
            else:
                # Create a visual indicator
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("AI Likelihood Score", f"{score:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Color-coded verdict
                    if score >= threshold:
                        st.markdown(f'<div class="warning-box"><h3>⚠️ Verdict: {details["verdict"]}</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-box"><h3>✅ Verdict: {details["verdict"]}</h3></div>', unsafe_allow_html=True)
                
                with col_b:
                    # Progress bar visualization
                    st.markdown("**Confidence Level:**")
                    st.progress(score/100)
                    
                    if show_details:
                        with st.expander("Technical Details"):
                            st.json(details)
        
        elif upload is not None:
            b = upload.read()
            name = (upload.name or "").lower()
            mime = (upload.type or "").lower()
            
            # detect type
            is_image = any(name.endswith(ext) for ext in (".png",".jpg",".jpeg",".bmp",".tiff",".webp")) or mime.startswith("image")
            is_audio = any(name.endswith(ext) for ext in (".wav",".mp3",".m4a",".flac",".ogg")) or mime.startswith("audio")
            is_video = any(name.endswith(ext) for ext in (".mp4",".mov",".avi",".mkv",".webm")) or mime.startswith("video")
            is_pdf = name.endswith(".pdf") or mime == "application/pdf"
            
            if is_image:
                with st.spinner("🔍 Analyzing image..."):
                    score, model_scores, details, pil = run_image_pipeline(b)
                
                st.image(pil, use_container_width=True)  # Fixed deprecated parameter
                
                if score is None:
                    st.markdown('<div class="warning-box">No valid model responses — check model logs in technical details.</div>', unsafe_allow_html=True)
                else:
                    col_a, col_b = st.columns([1, 2])
                    
                    with col_a:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("AI Likelihood Score", f"{score:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        verdict = "Likely AI-Generated" if score >= threshold else "Likely Authentic"
                        if score >= threshold:
                            st.markdown(f'<div class="warning-box"><h3>⚠️ Verdict: {verdict}</h3></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="success-box"><h3>✅ Verdict: {verdict}</h3></div>', unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown("**Confidence Level:**")
                        st.progress(score/100)
                
                if show_details:
                    with st.expander("Technical Details"):
                        st.json({"per_model_scores": model_scores, "model_outputs": details})
            
            elif is_audio:
                st.audio(b)
                
                with st.spinner("🔍 Analyzing audio..."):
                    score, details = run_audio_pipeline(b)
                
                if score is None:
                    if "error" in details:
                        st.markdown(f'<div class="error-box">{details["error"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">Audio model failed to return a score. See technical details.</div>', unsafe_allow_html=True)
                else:
                    col_a, col_b = st.columns([1, 2])
                    
                    with col_a:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("AI Likelihood Score", f"{score:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        verdict = "Likely AI-Generated" if score >= threshold else "Likely Authentic"
                        if score >= threshold:
                            st.markdown(f'<div class="warning-box"><h3>⚠️ Verdict: {verdict}</h3></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="success-box"><h3>✅ Verdict: {verdict}</h3></div>', unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown("**Confidence Level:**")
                        st.progress(score/100)
                
                if show_details:
                    with st.expander("Technical Details"):
                        st.json(details)
            
            elif is_video:
                st.video(b)
                
                with st.spinner("🔍 Analyzing video frames..."):
                    score, details = run_video_pipeline(b, max_frames=max_frames)
                
                if score is None:
                    st.markdown('<div class="warning-box">No valid frame scores. See technical details.</div>', unsafe_allow_html=True)
                else:
                    col_a, col_b = st.columns([1, 2])
                    
                    with col_a:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("AI Likelihood Score", f"{score:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        verdict = "Likely AI-Generated" if score >= threshold else "Likely Authentic"
                        if score >= threshold:
                            st.markdown(f'<div class="warning-box"><h3>⚠️ Verdict: {verdict}</h3></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="success-box"><h3>✅ Verdict: {verdict}</h3></div>', unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown("**Confidence Level:**")
                        st.progress(score/100)
                
                if show_details:
                    with st.expander("Technical Details"):
                        st.json(details)
            
            elif is_pdf:
                with st.spinner("🔍 Analyzing document..."):
                    res = analyze_pdf_bytes(b)
                
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Suspicion Score", f"{res['score']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if res['score'] >= threshold:
                        st.markdown(f'<div class="warning-box"><h3>⚠️ Verdict: {res["verdict"]}</h3></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-box"><h3>✅ Verdict: {res["verdict"]}</h3></div>', unsafe_allow_html=True)
                
                with col_b:
                    st.markdown("**Suspicion Level:**")
                    st.progress(res['score']/100)
                
                if show_details:
                    with st.expander("Technical Details"):
                        st.json(res["details"])
            
            else:
                st.markdown('<div class="warning-box">Unsupported file type. Try an image, audio, video, PDF, or paste text.</div>', unsafe_allow_html=True)
    
    elapsed = time.time() - start
    st.caption(f"Analysis completed in {elapsed:.2f} seconds")

# Footer
st.markdown("---")
st.markdown('<div class="footer">DeepGuard AI • Content Authenticity Verification • Made with Privacy in Mind</div>', unsafe_allow_html=True)