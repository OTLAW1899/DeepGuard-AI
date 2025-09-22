
## DeepGuard AI — ML-enhanced Deepfake Detection 

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff4b4b)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success)

DeepGuard AI is a **Streamlit-powered application** designed to detect and analyze AI-generated (deepfake) content across multiple modalities:  

***Text***  - ***Images***  - ***Audio***  - ***Video***

It uses **machine learning–enhanced detection** methods to evaluate the likelihood that content was AI-generated, giving users a clear and interactive way to verify authenticity.

---

### Features

- **Multi-modal support**: Analyze text, images, audio, and video.
- **File uploads**: Supports `.txt`, `.pdf`, `.docx`, `.rtf`, `.odt`, common image, audio, and video formats.  
- **Text extraction** from documents (PDF, Word, ODT, TXT).  
- **Interactive UI** with Streamlit.  
- **Real-time analysis progress** with progress bar.  
- **Configurable detection threshold** (decide when content is "likely AI-generated").  
- **Detailed JSON output** of analysis results.  

---

### Getting Started

#### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/deepguard-ai.git
cd deepguard-ai
````

#### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

#### 3. Install dependencies

The project dependencies are listed in `requirements.txt`:

```txt
streamlit
pymupdf
python-docx
odfpy
```

Install them with:

```bash
pip install -r requirements.txt
```

#### 4. Run the app

```bash
streamlit run main.py

![streamlit run](https://github.com/user-attachments/assets/ae0a0cdc-6c7c-4854-b6f7-3c501f6a58f5)

```

---

### Project Structure

```
deepguard-ai/
│── app.py                 # Main Streamlit application
│── utils.py               # Utility functions for analysis
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

---

### Functionalities

* Upload a **PDF or DOCX file** to analyze text authenticity.
* Upload an **image, audio, or video** to analyze multimedia authenticity.
* Paste **raw text** directly for fast detection.
* View **detailed JSON results** in the expandable section.

  <img width="960" height="405" alt="90 5% AI-GENERATED AUDIO" src="https://github.com/user-attachments/assets/6ac5bbdc-661e-4b01-9b0c-cefe6a185d04" />
  <img width="956" height="432" alt="AI GENERATED CONTENT" src="https://github.com/user-attachments/assets/b24c9f02-0a0e-4d4d-891b-7752d7bac127" />
  <img width="960" height="421" alt="AI GENERATED TEXT" src="https://github.com/user-attachments/assets/7150b25a-2015-4d61-aa69-032a90bbe767" />
  <img width="954" height="415" alt="REAL PICTURE" src="https://github.com/user-attachments/assets/484f2785-e91d-4c43-927b-5415629e462c" />
  





---

### Author


***Olatunji Lawal***

Cybersecurity Analyst

---

### License

This project is licensed under the **MIT License** — feel free to use, modify, and share. 
