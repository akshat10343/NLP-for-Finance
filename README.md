# NLP for Finance

Sentiment classification of financial news sentences (negative, neutral or positive). Three notebooks build up from a simple baseline to a fine-tuned transformer.

This was a guided project in Inspirit AI's AI Scholars program (NLP + Finance). The notebooks run in Google Colab.

| Notebook | What it covers |
|---|---|
| `NLPForFinance_Section1.ipynb` | Data exploration (label balance, word clouds), text cleaning (punctuation and stopword removal), bag-of-words with `CountVectorizer`, a logistic regression baseline, and a confusion matrix |
| `NLPForFinance_Section2.ipynb` | A Keras LSTM: tokenized, padded sequences (up to 256 tokens, 1,000-word vocabulary), an embedding layer, `LSTM(100)` and a softmax output, plus loss and accuracy curves |
| `NLPForFinance_Section3.ipynb` | Fine-tuning `bert-base-uncased` (`BertForSequenceClassification`, PyTorch and Hugging Face Transformers) with AdamW |

## Data

2,037 training sentences and 227 test sentences, each labeled negative (0), neutral (1) or positive (2). The setup cells fetch the data with `wget`.

## Results

The fine-tuned BERT model reached **96.92% accuracy on the 227-sentence test set**. Validation accuracy rose from 0.80 to 0.97 during training. Both numbers are saved outputs in Section 3.

## Run it

Open a notebook in Google Colab and run the cells in order. Section 3 trains much faster on a GPU runtime (Runtime → Change runtime type → GPU).

**Stack:** Python, pandas, NLTK, scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers
