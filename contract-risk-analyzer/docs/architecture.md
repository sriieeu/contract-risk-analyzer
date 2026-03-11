# Architecture Overview

## Pipeline Flow

```
PDF Upload
    │
    ▼
PDFExtractor (PyMuPDF)
    │  Extracts raw text, cleans headers/footers
    ▼
ClauseSegmenter (spaCy + regex)
    │  Splits into numbered/paragraph clauses
    ▼
CUADModel (BERT fine-tuned)
    │  Classifies each clause into 32 types
    ▼
RiskScorer (rule engine)
    │  Scores 0-100, detects flags, generates explanations
    ▼
SHAPExplainer
    │  Token-level importance, highlighted HTML
    ▼
Streamlit UI (5 tabs)
    Upload → Clause Review → SHAP → Redlines → Dashboard
```

## Module Responsibilities

| Module | Input | Output |
|--------|-------|--------|
| `PDFExtractor` | PDF bytes | `ExtractedDocument` |
| `ClauseSegmenter` | Contract text | `SegmentationResult` (list of `Clause`) |
| `CUADModel` | `Clause` list | `ClassificationResult` list |
| `RiskScorer` | `ClassificationResult` list | `ContractRisk` |
| `SHAPExplainer` | Clause text + type | `ExplanationResult` (HTML) |
| `RedlineDiff` | Two contract texts | `DiffResult` (HTML) |
