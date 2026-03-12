Organizations deal with a large number of legal contracts such as vendor agreements, employment contracts, NDAs, and licensing agreements. Reviewing these documents manually is time-consuming and requires legal expertise because each contract contains multiple clauses that may introduce legal, financial, or compliance risks.

The Contract Risk Analyzer is a legal document analysis system designed to automate the identification and evaluation of risky clauses in contracts. It uses Natural Language Processing (NLP) techniques and a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model trained on the CUAD (Contract Understanding Atticus Dataset) to classify contract clauses and assess potential risk levels.

The system extracts text from contract PDFs, segments them into clauses, classifies each clause into predefined legal categories, calculates a risk score, and explains why a clause is considered risky. The application also provides tools for comparing contract versions and visualizing model explanations.

Objectives
The main objectives of the Contract Risk Analyzer system are:
Automate the extraction and analysis of legal clauses from contract documents.
Classify clauses into standard legal categories based on the CUAD benchmark.
Identify potential legal risks within specific clauses.
Provide explainable AI outputs showing why the model flagged a clause.
Allow contract comparison through a redline diff viewer.
Enable quick review of contracts through an interactive web interface

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

Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Optional: Fine-tune on CUAD dataset
python src/classification/train_cuad.py

# Run the app
streamlit run src/ui/app.py

CUAD Dataset

The Contract Understanding Atticus Dataset (CUAD) (https://www.atticusprojectai.org/cuad) contains 510 contracts with 13,000+ expert annotations across 41 legal clause categories. We fine-tune `bert-base-uncased` on this benchmark for clause classification.

Tech Stack
- **PDF Parsing**: PyMuPDF (fitz)
- **NLP Pipeline**: spaCy (en_core_web_sm)
- **Model**: BERT fine-tuned on CUAD (HuggingFace Transformers)
- **Explainability**: SHAP (transformers pipeline explainer)
- **UI**: Streamlit
- **Diff**: difflib + custom React-style renderer



Web Preview:
<img width="1808" height="802" alt="image" src="https://github.com/user-attachments/assets/b0c8eb87-aa71-4f99-b5eb-017ee199077e" />
<img width="1441" height="861" alt="image" src="https://github.com/user-attachments/assets/90216d69-c4d0-4182-9541-5161981603b4" />
<img width="1473" height="770" alt="image" src="https://github.com/user-attachments/assets/5d544e8d-aca7-4ec7-a613-2838b05c206f" />
<img width="1420" height="744" alt="image" src="https://github.com/user-attachments/assets/d40c185c-9161-4486-9bde-af09fdb91c29" />


<img width="1417" height="874" alt="image" src="https://github.com/user-attachments/assets/3d83fb30-c716-419b-864a-4504464cd1c5" />

<img width="1478" height="777" alt="image" src="https://github.com/user-attachments/assets/fdecfe93-db14-4369-abad-b4dd9a985127" />
