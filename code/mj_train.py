# ------------------------Libraries------------------------------------
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import pickle
from joblib import dump
from xgboost import XGBClassifier

# Install required packages (only runs once)
os.system('pip3 install pandas scikit-learn openpyxl xgboost nltk')

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

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
    df = pd.read_excel(path)
    df['Text'] = df['Text'].apply(preprocess_text)
    df['Text'] = df['Text'].apply(remove_stopwords)
    df['Text'] = df['Text'].apply(lemmatize_text)
    return df

def split(df, split='test'):
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == split]
    return train_df, test_df

def model_traning(train_df, label_encoder):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['Text'])

    train_df['Label_enc'] = label_encoder.fit_transform(train_df['Label'])
    sample_weights = compute_sample_weight(class_weight='balanced', y=train_df['Label_enc'])

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
        learning_rate=0.1,
        max_depth=4,              
        n_estimators=100,         
        subsample=0.8,
        colsample_bytree=0.8
    )

    print("🚀 Starting training...")
    model.fit(
        X_train,
        train_df['Label_enc'],
        sample_weight=sample_weights,
        eval_set=[(X_train, train_df['Label_enc'])],
        early_stopping_rounds=10, 
        verbose=True            
    )
    print("✅ Training complete.")
    
    return model, vectorizer

def model_testing(model, vectorizer, test_df):
    X_test = vectorizer.transform(test_df['Text'])
    return model.predict(X_test)

def metrics(predictions, test_df):
    kappa = cohen_kappa_score(test_df['Label'], predictions)
    f1 = f1_score(test_df['Label'], predictions, average='macro')
    return float(0.5 * kappa + 0.5 * f1)

def save_model(model):
    with open(f'{nickname}_model.pickle', 'wb') as f:
        pickle.dump(model, f)

def save_vectorizer(vectorizer, file_name):
    dump(vectorizer, file_name)

def save_label_encoder(label_encoder, file_name):
    dump(label_encoder, file_name)

# ---------------------------------------------------------------------

def main():
    df = read_data(path="/home/ubuntu/NLP_Exam1/Data/train.xlsx")
    train_df, test_df = split(df, split='test')
    label_encoder = LabelEncoder()
    model, vectorizer = model_traning(train_df, label_encoder)
    predictions = model_testing(model, vectorizer, test_df)
    predictions = label_encoder.inverse_transform(predictions)
    metric_score = metrics(predictions, test_df)
    print(f"📊 Final Metric Score: {metric_score}")
    save_model(model)
    save_vectorizer(vectorizer, f'{nickname}_vectorizer.joblib')
    save_label_encoder(label_encoder, f'{nickname}_labelencoder.joblib')

# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()
