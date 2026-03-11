#  Contract Risk Analyzer

A production-grade legal contract analysis system using NLP and fine-tuned BERT on the CUAD dataset.

## Features

- **25+ Clause Types**: CUAD benchmark clause classification
- **Risk Scoring**: Per-clause risk scores with plain-English explanations
- **SHAP Explainability**: Understand what drives each risk flag
- **Redline Diff Viewer**: Side-by-side contract comparison
- **PDF Extraction**: Automatic clause segmentation via PyMuPDF + spaCy

## Architecture

```
contract-risk-analyzer/
├── src/
│   ├── extraction/        # PDF parsing + clause segmentation
│   │   ├── pdf_extractor.py
│   │   └── clause_segmenter.py
│   ├── classification/    # BERT fine-tuned on CUAD
│   │   ├── cuad_model.py
│   │   └── clause_classifier.py
│   ├── risk/              # Risk scoring engine
│   │   ├── risk_scorer.py
│   │   └── risk_rules.py
│   ├── explainability/    # SHAP integration
│   │   └── shap_explainer.py
│   └── ui/                # Streamlit app
│       └── app.py
├── data/
│   └── cuad_labels.json   # CUAD clause taxonomy
├── models/                # Fine-tuned model checkpoints
├── tests/
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Optional: Fine-tune on CUAD dataset
python src/classification/train_cuad.py

# Run the app
streamlit run src/ui/app.py
```

## CUAD Dataset

The [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad) contains 510 contracts with 13,000+ expert annotations across 41 legal clause categories. We fine-tune `bert-base-uncased` on this benchmark for clause classification.

## Tech Stack

- **PDF Parsing**: PyMuPDF (fitz)
- **NLP Pipeline**: spaCy (en_core_web_sm)
- **Model**: BERT fine-tuned on CUAD (HuggingFace Transformers)
- **Explainability**: SHAP (transformers pipeline explainer)
- **UI**: Streamlit
- **Diff**: difflib + custom React-style renderer

- 
