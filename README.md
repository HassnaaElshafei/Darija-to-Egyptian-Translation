# Darija to Egyptian Arabic Translation

An interactive Moroccan Darija to Egyptian Arabic translation system with two
inference pipelines, exposed through a Streamlit web interface for easy testing
and comparison.

---
## Live Demo (Streamlit)
Try the app here:
https://darija-to-egyptian-translation-jjthztbcxdtr8g5nfrn82u.streamlit.app/

## Features

- Two translation approaches:
  - Direct translation using a fine-tuned NLLB model
  - Two-step translation using AraT5:
    - Darija to Modern Standard Arabic (MSA)
    - MSA to Egyptian Arabic using a LoRA-adapted AraT5v2 model
- Streamlit-based web interface
- Lightweight and modular design using LoRA adapters
- Fully based on open-source models

---

## Models Used

### Direct Translation (NLLB)
- Base model: facebook/nllb-200-distilled-1.3B
- Fine-tuned with LoRA for Darija to Egyptian Arabic

### Two-step Translation (AraT5)
- Darija to MSA: Saidtaoussi/AraT5_Darija_to_MSA
- MSA to Egyptian: UBC-NLP/AraT5v2-base-1024 with a LoRA adapter

This repository contains code only. Model adapters are downloaded automatically
from Hugging Face at runtime.

---

## Project Structure

.
├── app.py
├── smoke_test.py
├── requirements.txt
└── README.md

---

## Setup

Clone the repository and create a virtual environment:

git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## Smoke Test (Optional)

Run the smoke test to verify that base models and adapters load correctly:

python smoke_test.py

---

## Run the Application

Start the Streamlit app:

streamlit run app.py

The application will open in your browser and allow you to enter Darija text,
select a translation method, and view the Egyptian Arabic output.

---

## Deployment

This project can be deployed using:
- Streamlit Community Cloud
- Hugging Face Spaces

---

## License

The code in this repository is released under the Apache 2.0 license.

Model weights and adapters are subject to their original licenses.

---

## Author

Hassnaa Elshafei
