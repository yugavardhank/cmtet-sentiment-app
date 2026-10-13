# CMTet Sentiment Analyzer

Telugu-English code-mixed sentiment analysis (NEG / NTL / POS) using fine-tuned MuRIL and XLM-RoBERTa.

## Dataset

Kusampudi, S., Parimi, P., & Bandyopadhyay, S. (2021). **Sentiment Analysis in Code-Mixed Telugu-English Texts with Unsupervised Data Normalization**. In *Proceedings of RANLP 2021*. [ACL Anthology](https://aclanthology.org/2021.ranlp-1.89/)

## Project Structure

```
cmtet-sentiment-app/
  data/raw/codemix_sentiment_data.txt  # raw CMTet corpus
  data/processed/                      # generated clean.csv, splits.json
  src/
    config.py           # all paths, labels, hyperparameters
    data.py             # parse, dedupe, clean, CMI, split
    baseline.py         # TF-IDF + LinearSVC (M1)
    train_transformer.py # XLM-R / MuRIL fine-tuning
    evaluate.py         # metrics, CMI analysis, confusion matrix
    predict.py          # inference class
  app/
    main.py             # FastAPI backend
    static/index.html   # frontend
  models/best/          # best checkpoint (gitignored)
  results/              # metrics JSON, plots, error_samples.csv
  tests/                # pytest suite
  Dockerfile
  requirements.txt
```

## Setup & API Key

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional but Recommended) Configure Google Gemini API:
Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and add it to your `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
> **Note**: If `GEMINI_API_KEY` is not provided, the app automatically runs in fallback mode using the fast TF-IDF baseline, so the application runs seamlessly out-of-the-box in all environments.

## Module 2 - Data Pipeline

```bash
python src/data.py
```

Produces `data/processed/clean.csv` and `data/processed/splits.json`.

## Module 4 - Baseline

```bash
python src/baseline.py
```

## Module 6 - Evaluation

```bash
python src/evaluate.py
```

Generates results table, CMI breakdown (`results/cmi_breakdown.png`), confusion matrix, and error analysis.

## Module 7 - Inference Test

```bash
python src/predict.py
```

Runs a 10-sentence smoke test on Telugu-English code-mixed comments.

## Running the Web App (Modules 8 & 9)

```bash
uvicorn app.main:app --reload --port 7860
```
Open [http://localhost:7860](http://localhost:7860) to use the interactive frontend.

## Running Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t cmtet-sentiment .
docker run -e GEMINI_API_KEY=<your_key> -p 7860:7860 cmtet-sentiment
```

