import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import json
import sqlite3
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class EnhancedSentimentModelPipeline:
    """簡化版：相容 app_enhanced.py，保留核心功能"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.best_model = None
        self.model_performances = {}
        self.db_path = 'sentiment_analysis.db'
        
    def setup_database(self):
        """建立相容表結構"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                model_name TEXT NOT NULL,
                latency REAL DEFAULT 0.0,
                user_feedback INTEGER DEFAULT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision_pos REAL NOT NULL,
                precision_neg REAL NOT NULL,
                recall_pos REAL NOT NULL,
                recall_neg REAL NOT NULL,
                f1_score REAL NOT NULL,
                auc_score REAL NOT NULL,
                training_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_a TEXT NOT NULL,
                model_b TEXT NOT NULL,
                text TEXT NOT NULL,
                prediction_a INTEGER NOT NULL,
                prediction_b INTEGER NOT NULL,
                confidence_a REAL NOT NULL,
                confidence_b REAL NOT NULL,
                user_feedback INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ 數據庫表建立完成")
    
    def load_and_preprocess_data(self):
        print("📚 載入電影評論數據...")
        try:
            nltk.download("movie_reviews", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            print("✓ NLTK 數據準備完成")
        except Exception as e:
            print(f"⚠ NLTK 數據下載失敗: {e}")
        
        documents = [(list(movie_reviews.words(fileid)), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]
        
        random.seed(42)
        np.random.seed(42)
        random.shuffle(documents)
        
        texts, labels = [], []
        for words, label in documents:
            text = " ".join(words).lower()
            if len(text.split()) > 10:
                texts.append(text)
                labels.append(1 if label == "pos" else 0)
        
        print(f"資料量: {len(texts)} 筆 (正面 {sum(labels)}, 負面 {len(labels) - sum(labels)})")
        return texts, labels
    
    def prepare_features(self, texts, labels):
        print("🔧 特徵工程 (TF-IDF)...")
        # 先分割文字，再以訓練集擬合向量化器，避免資料洩漏
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.8,
            sublinear_tf=True,
            norm='l2'
        )
        X_train = self.vectorizer.fit_transform(X_train_texts)
        X_test = self.vectorizer.transform(X_test_texts)
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        print("🤖 初始化模型...")
        self.models = {
            'Logistic_Regression': LogisticRegression(max_iter=3000, random_state=42, solver='liblinear', class_weight='balanced'),
            'Random_Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced', kernel='linear')
        }
        return self.models
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        print("🏃‍♂️ 訓練與評估...")
        results = {}
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                accuracy = model.score(X_test, y_test)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_macro')
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'precision_pos': report.get('1', {}).get('precision', 0),
                    'precision_neg': report.get('0', {}).get('precision', 0),
                    'recall_pos': report.get('1', {}).get('recall', 0),
                    'recall_neg': report.get('0', {}).get('recall', 0),
                    'f1_score': report.get('macro avg', {}).get('f1-score', 0),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std())
                }
                print(f"{name}: acc={accuracy:.4f} auc={auc_score:.4f} f1={results[name]['f1_score']:.4f}")
            except Exception as e:
                print(f"  ❌ 訓練 {name} 失敗: {e}")
        self.model_performances = results
        return results
    
    def select_best_model(self, results):
        print("🏆 選擇最佳模型 (綜合: acc+f1+auc)/3 ...")
        best_score = -1
        best_model_name = None
        for name, metrics in results.items():
            composite = (metrics['accuracy'] + metrics['f1_score'] + metrics['auc_score']) / 3
            if composite > best_score:
                best_score = composite
                best_model_name = name
        if best_model_name:
            self.best_model = results[best_model_name]['model']
            print(f"✓ 最佳模型: {best_model_name} (score={best_score:.4f})")
        else:
            print("❌ 未能選出最佳模型")
        return best_model_name, self.best_model
    
    def save_model_performance_to_db(self, results):
        print("💾 保存模型性能到 DB ...")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for name, m in results.items():
                cursor.execute('''
                    INSERT INTO model_performance 
                    (model_name, accuracy, precision_pos, precision_neg, recall_pos, recall_neg, f1_score, auc_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    name,
                    float(m['accuracy']),
                    float(m['precision_pos']),
                    float(m['precision_neg']),
                    float(m['recall_pos']),
                    float(m['recall_neg']),
                    float(m['f1_score']),
                    float(m['auc_score'])
                ))
            conn.commit()
            conn.close()
            print("✓ 已保存")
        except Exception as e:
            print(f"❌ 保存失敗: {e}")
    
    def save_models_for_app(self, best_model_name):
        print("💾 保存模型/向量化器/報告...")
        try:
            if self.best_model:
                pickle.dump(self.best_model, open("best_model.pkl", "wb"))
                pickle.dump(self.best_model, open("model.pkl", "wb"))
            if self.vectorizer:
                pickle.dump(self.vectorizer, open("vectorizer.pkl", "wb"))
            if self.models:
                pickle.dump(self.models, open("all_models.pkl", "wb"))
            if self.model_performances:
                serializable = {}
                for name, metrics in self.model_performances.items():
                    serializable[name] = {
                        k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                        for k, v in metrics.items() if k not in ['model']
                    }
                report_data = {
                    'best_model': best_model_name,
                    'performances': serializable,
                    'training_date': datetime.now().isoformat(),
                    'model_files': {
                        'best_model': 'best_model.pkl',
                        'vectorizer': 'vectorizer.pkl',
                        'all_models': 'all_models.pkl'
                    }
                }
                with open("model_performance_report.json", "w", encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
            print("✓ 檔案已保存")
        except Exception as e:
            print(f"❌ 保存失敗: {e}")
            return False
        return True
    
    def run_full_pipeline(self):
        print("="*60)
        print("🚀 開始訓練 (簡化版)")
        print("="*60)
        try:
            self.setup_database()
            texts, labels = self.load_and_preprocess_data()
            X_train, X_test, y_train, y_test = self.prepare_features(texts, labels)
            self.initialize_models()
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
            if not results:
                print("❌ 沒有成功訓練的模型")
                return False
            best_model_name, _ = self.select_best_model(results)
            if not best_model_name:
                return False
            self.save_model_performance_to_db(results)
            if not self.save_models_for_app(best_model_name):
                return False
            print("\n✅ 訓練完成，已可啟動 app_enhanced.py")
            return True
        except Exception as e:
            print(f"❌ 訓練失敗: {e}")
            return False


def main():
    pipeline = EnhancedSentimentModelPipeline()
    success = pipeline.run_full_pipeline()
    if success:
        print("最佳模型/向量化器與報告已產生。")

if __name__ == "__main__":
    main()