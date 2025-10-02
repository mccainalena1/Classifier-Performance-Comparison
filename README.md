# Classifier Performance Comparison

Supervised classifier benchmark on the **20 Newsgroups** dataset. 
Trains and evaluates multiple models and reports metrics and figures.

## Repository Structure
```
.
├── src/
│   └── assignment2.py
├── data/
│   └── 2Newsgroups/
├── results/
│   ├── Evaluation.pdf
│   ├── Figure_1.png
│   ├── KNN.txt
│   ├── NB.txt
│   └── SVM.txt
├── docs/
│   ├── asg2-2023.pdf
│   └── asg2.pdf
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python src/assignment2.py
```

## Data
The project uses the preprocessed **20 Newsgroups** files located in `data/2Newsgroups/`.

## Results
Evaluation artifacts are in `results/`:
- `Evaluation.pdf` – summary write‑up
- `KNN.txt`, `NB.txt`, `SVM.txt` – classifier metrics
- `Figure_1.png` – visualization

## License
MIT © 2025 Alena McCain
