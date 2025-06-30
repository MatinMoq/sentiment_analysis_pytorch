# Sentiment Analyzer (PyTorch)

A simple and production-ready end-to-end sentiment analysis project using LSTM and PyTorch, based on the IMDB dataset.

---

## Project Structure

```plaintext
sentiment-analyzer-pytorch/
├── data/
├── models/
│   └── lstm_model.py
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app/
│   └── gradio_app.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
└── README.md


## How to run

### 1. Install requirements

pip install -r requirements.txt

### 2. Train model
python src/train.py

### 3. Evaluate model

python src/evaluate.py
### 4. Predict sentiment

python src/predict.py
### 5. Run Gradio app

python app/gradio_app.py
