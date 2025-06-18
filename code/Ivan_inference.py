# -*- coding: utf-8 -*-
# ------------------------Libraries------------------------------------
import argparse
import os
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import cohen_kappa_score, f1_score
import pickle
from joblib import load

# Install required packages (if needed)
os.system('pip3 install pandas xgboost scikit-learn openpyxl nltk')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

nickname = 'mj'

# ------------------------Text Preprocessing---------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(word) for word in words])
# ---------------------------------------------------------------------

def read_data(path):
    df = pd.read_excel(path, engine='openpyxl')
    df['Text'] = df['Text'].apply(preprocess_text)
    df['Text'] = df['Text'].apply(remove_stopwords)
    df['Text'] = df['Text'].apply(lemmatize_text)
    return df

def split_data(df, split='test'):
    return df[df['split'] == split]

def load_model(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_vectorizer(file_name):
    return load(file_name)

def load_labelencoder(file_name):
    return load(file_name)

def inference(test_df, vectorizer, label_encoder, name):
    X_test = vectorizer.transform(test_df['Text'])
    model = load_model(name)
    preds_encoded = model.predict(X_test)
    preds_decoded = label_encoder.inverse_transform(preds_encoded)
    return preds_decoded, preds_encoded

def metrics(predictions, test_df):
    kappa = cohen_kappa_score(test_df['Label'], predictions)
    f1 = f1_score(test_df['Label'], predictions, average='macro')
    return 0.5 * kappa + 0.5 * f1

def main(data_path, train_df_name, model_path, split):
    print("📥 Reading data...")
    df = read_data(os.path.join(data_path, train_df_name))
    test_df = split_data(df, split)

    print("📦 Loading models...")
    label_encoder = load_labelencoder(f'{model_path}/{nickname}_labelencoder.joblib')
    vectorizer = load_vectorizer(f'{model_path}/{nickname}_vectorizer.joblib')

    print("🔍 Running inference...")
    predictions, _ = inference(test_df, vectorizer, label_encoder, f'{model_path}/{nickname}_model.pickle')

    print("💾 Saving predictions...")
    np.save(f'{model_path}/{nickname}-{split}_predictions.npy', predictions)

    print("📊 Calculating metric score...")
    score = metrics(predictions, test_df)
    print(f'✅ Metric score: {score:.4f}')

# ---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run XGBoost text inference')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--train_df', type=str, required=True, help='Filename for training/testing dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model and artifacts')
    parser.add_argument('--split', type=str, default='test', help='Split type (train/test)')
    args = parser.parse_args()

    main(args.data_path, args.train_df, args.model_path, args.split)
