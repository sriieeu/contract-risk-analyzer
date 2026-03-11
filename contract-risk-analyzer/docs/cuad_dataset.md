# CUAD Dataset Reference

## Overview
The Contract Understanding Atticus Dataset (CUAD) is a legal AI benchmark
published by The Atticus Project and presented at NeurIPS 2021.

- **510** commercial contracts
- **13,000+** expert annotations
- **41** clause categories
- Annotated by legal professionals

## Citation
```
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Hendrycks, Dan and others},
  journal={NeurIPS 2021},
  year={2021}
}
```

## Links
- Dataset: https://huggingface.co/datasets/cuad
- Paper: https://arxiv.org/abs/2103.06268
- Atticus Project: https://www.atticusprojectai.org/cuad
- Pre-trained model: https://huggingface.co/theatticusproject/cuad-qa

## Fine-tuning
See `src/classification/train_cuad.py` for the full training script.
Recommended hardware: GPU with 16GB+ VRAM (e.g. A10, T4, RTX 3080).

```bash
make train          # Full 3-epoch training
make train-fast     # Quick 1-epoch run
```
