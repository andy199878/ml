import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
import pickle
import json
import sqlite3
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

class EnhancedSentimentModelPipeline:
    """完全相容新版app_enhanced.py的情感分析模型管道"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.best_model = None
        self.model_performances = {}
        self.db_path = 'sentiment_analysis.db'
        
    def setup_database(self):
        """建立完全相容的數據庫表結構"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 預測記錄表 - 添加latency和user_feedback欄位
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
        
        # 模型性能表
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
        
        # A/B測試表
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
        print("✓ 數據庫表建立完成 (相容app_enhanced.py)")
    
    def load_and_preprocess_data(self):
        """載入並預處理數據"""
        print("📚 載入電影評論數據...")
        
        # 下載NLTK數據
        try:
            nltk.download("movie_reviews", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            print("✓ NLTK 數據準備完成")
        except Exception as e:
            print(f"⚠ NLTK 數據下載失敗: {e}")
        
        # 載入數據
        documents = [(list(movie_reviews.words(fileid)), category)
                     for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]
        
        random.seed(42)
        np.random.seed(42)
        random.shuffle(documents)
        
        # 預處理
        texts = []
        labels = []
        
        for words, label in documents:
            text = " ".join(words).lower()
            # 過濾太短的文本
            if len(text.split()) > 10:
                texts.append(text)
                labels.append(1 if label == "pos" else 0)
        
        print(f"資料量: {len(texts)} 筆")
        print(f"正面: {sum(labels)} 筆 ({sum(labels)/len(labels)*100:.1f}%)")
        print(f"負面: {len(labels) - sum(labels)} 筆 ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
        
        return texts, labels
    
    def prepare_features(self, texts, labels):
        """特徵工程"""
        print("🔧 進行特徵工程...")
        
        # 使用TF-IDF向量化 - 針對app性能優化
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.8,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        X = self.vectorizer.fit_transform(texts)
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"訓練集: {X_train.shape[0]} 筆")
        print(f"測試集: {X_test.shape[0]} 筆")
        print(f"特徵維度: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """初始化多種模型 - 針對生產環境優化"""
        print("🤖 初始化多種機器學習模型...")
        
        self.models = {
            'best_model': LogisticRegression(  # 主要模型
                max_iter=3000,
                random_state=42,
                class_weight='balanced',
                solver='liblinear'
            ),
            'Logistic_Regression': LogisticRegression(
                max_iter=3000,
                random_state=42,
                class_weight='balanced',
                solver='liblinear'
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1  # 加速訓練
            ),
            'SVM': SVC(
                probability=True,
                random_state=42,
                class_weight='balanced',
                kernel='linear'  # 更快的線性核
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1  # 加速訓練
            )
        }
        
        return self.models
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """訓練並評估所有模型"""
        print("🏃‍♂️ 開始訓練和評估模型...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n訓練 {name}...")
            
            try:
                # 訓練模型
                model.fit(X_train, y_train)
                
                # 預測
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # 評估指標
                accuracy = model.score(X_test, y_test)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # 分類報告
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                # 交叉驗證
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
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"  準確率: {accuracy:.4f}")
                print(f"  AUC分數: {auc_score:.4f}")
                print(f"  F1分數: {report.get('macro avg', {}).get('f1-score', 0):.4f}")
                print(f"  交叉驗證: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"  ❌ 訓練 {name} 失敗: {e}")
                continue
        
        self.model_performances = results
        return results
    
    def select_best_model(self, results):
        """選擇最佳模型"""
        print("🏆 選擇最佳模型...")
        
        # 綜合評分 (準確率 + F1分數 + AUC分數) / 3
        best_score = 0
        best_model_name = None
        
        for name, metrics in results.items():
            if name == 'best_model':  # 跳過重複的best_model
                continue
                
            composite_score = (metrics['accuracy'] + metrics['f1_score'] + metrics['auc_score']) / 3
            print(f"{name}: 綜合評分 {composite_score:.4f}")
            
            if composite_score > best_score:
                best_score = composite_score
                best_model_name = name
        
        if best_model_name:
            self.best_model = results[best_model_name]['model']
            # 同時更新best_model條目
            self.models['best_model'] = self.best_model
            print(f"✓ 最佳模型: {best_model_name} (評分: {best_score:.4f})")
            return best_model_name, self.best_model
        else:
            print("❌ 未找到有效的最佳模型")
            return None, None
    
    def save_model_performance_to_db(self, results):
        """將模型性能保存到數據庫"""
        print("💾 保存模型性能到數據庫...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for name, metrics in results.items():
                cursor.execute('''
                    INSERT INTO model_performance 
                    (model_name, accuracy, precision_pos, precision_neg, 
                     recall_pos, recall_neg, f1_score, auc_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    name,
                    float(metrics['accuracy']),
                    float(metrics['precision_pos']),
                    float(metrics['precision_neg']),
                    float(metrics['recall_pos']),
                    float(metrics['recall_neg']),
                    float(metrics['f1_score']),
                    float(metrics['auc_score'])
                ))
            
            conn.commit()
            conn.close()
            print("✓ 模型性能已保存到數據庫")
            
        except Exception as e:
            print(f"❌ 保存到數據庫失敗: {e}")
    
    def create_visualizations(self, results, y_test):
        """創建視覺化圖表"""
        print("📊 創建模型比較視覺化...")
        
        try:
            # 確保static目錄存在
            os.makedirs('static', exist_ok=True)
            
            # 過濾有效結果
            valid_results = {name: metrics for name, metrics in results.items() 
                           if 'y_pred' in metrics and metrics['y_pred'] is not None}
            
            if not valid_results:
                print("⚠ 沒有有效的結果用於視覺化")
                return
            
            # 1. 模型性能比較
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('模型性能比較', fontsize=16)
            
            models = list(valid_results.keys())
            accuracy_scores = [valid_results[name]['accuracy'] for name in models]
            f1_scores = [valid_results[name]['f1_score'] for name in models]
            auc_scores = [valid_results[name]['auc_score'] for name in models]
            
            # 準確率比較
            axes[0, 0].bar(models, accuracy_scores, color='skyblue')
            axes[0, 0].set_title('準確率比較')
            axes[0, 0].set_ylabel('準確率')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # F1分數比較
            axes[0, 1].bar(models, f1_scores, color='lightgreen')
            axes[0, 1].set_title('F1分數比較')
            axes[0, 1].set_ylabel('F1分數')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # AUC分數比較
            axes[1, 0].bar(models, auc_scores, color='orange')
            axes[1, 0].set_title('AUC分數比較')
            axes[1, 0].set_ylabel('AUC分數')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 綜合評分
            composite_scores = [(accuracy_scores[i] + f1_scores[i] + auc_scores[i])/3 
                              for i in range(len(models))]
            axes[1, 1].bar(models, composite_scores, color='purple')
            axes[1, 1].set_title('綜合評分')
            axes[1, 1].set_ylabel('綜合評分')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('static/model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 混淆矩陣 - 只顯示前4個模型
            display_models = list(valid_results.items())[:4]
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('各模型混淆矩陣', fontsize=16)
            
            for i, (name, metrics) in enumerate(display_models):
                row, col = i // 2, i % 2
                cm = confusion_matrix(y_test, metrics['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
                axes[row, col].set_title(f'{name}')
                axes[row, col].set_xlabel('預測值')
                axes[row, col].set_ylabel('實際值')
            
            # 如果模型少於4個，隱藏多餘的子圖
            for i in range(len(display_models), 4):
                row, col = i // 2, i % 2
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('static/confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ 視覺化圖表已生成並保存到 static/ 目錄")
            
        except Exception as e:
            print(f"❌ 視覺化生成失敗: {e}")
    
    def save_models_for_app(self, best_model_name):
        """保存模型文件以供app_enhanced.py使用"""
        print("💾 保存模型和向量化器...")
        
        try:
            # 保存最佳模型 (app_enhanced.py優先載入)
            if self.best_model:
                pickle.dump(self.best_model, open("best_model.pkl", "wb"))
                print("✓ 最佳模型已保存為 best_model.pkl")
            
            # 保存向量化器
            if self.vectorizer:
                pickle.dump(self.vectorizer, open("vectorizer.pkl", "wb"))
                print("✓ 向量化器已保存為 vectorizer.pkl")
            
            # 保存所有模型 (供A/B測試使用)
            if self.models:
                pickle.dump(self.models, open("all_models.pkl", "wb"))
                print("✓ 所有模型已保存為 all_models.pkl")
            
            # 相容性：也保存為原來的名稱
            if self.best_model:
                pickle.dump(self.best_model, open("model.pkl", "wb"))
                print("✓ 相容性模型已保存為 model.pkl")
            
            # 保存模型性能報告 (供儀表板使用)
            if self.model_performances:
                serializable_performances = {}
                for name, metrics in self.model_performances.items():
                    serializable_performances[name] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in metrics.items()
                        if k not in ['model', 'y_pred', 'y_pred_proba']
                    }
                
                report_data = {
                    'best_model': best_model_name,
                    'performances': serializable_performances,
                    'training_date': datetime.now().isoformat(),
                    'model_files': {
                        'best_model': 'best_model.pkl',
                        'vectorizer': 'vectorizer.pkl',
                        'all_models': 'all_models.pkl'
                    }
                }
                
                with open("model_performance_report.json", "w", encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
                
                print("✓ 性能報告已保存為 model_performance_report.json")
            
        except Exception as e:
            print(f"❌ 模型保存失敗: {e}")
            return False
        
        return True
    
    def test_app_compatibility(self):
        """測試與app_enhanced.py的相容性"""
        print("🧪 測試與app_enhanced.py的相容性...")
        
        test_cases = [
            "This movie is absolutely amazing and fantastic!",
            "Terrible film, complete waste of time.",
            "Pretty decent movie, not bad at all.",
            "I hate this boring and awful film."
        ]
        
        try:
            # 測試模型載入
            if os.path.exists("best_model.pkl") and os.path.exists("vectorizer.pkl"):
                model = pickle.load(open("best_model.pkl", "rb"))
                vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
                print("✓ 模型文件載入成功")
                
                # 測試預測功能
                for i, text in enumerate(test_cases, 1):
                    vec = vectorizer.transform([text])
                    prediction = model.predict(vec)[0]
                    probability = model.predict_proba(vec)[0]
                    confidence = max(probability)
                    
                    sentiment = "positive" if prediction == 1 else "negative"
                    print(f"  測試 {i}: {sentiment} (信心度: {confidence:.3f})")
                
                print("✓ 預測功能測試通過")
                return True
                
            else:
                print("❌ 模型文件不存在")
                return False
                
        except Exception as e:
            print(f"❌ 相容性測試失敗: {e}")
            return False
    
    def run_full_pipeline(self):
        """執行完整的訓練管道"""
        print("="*70)
        print("🚀 開始執行增強版情感分析模型訓練管道")
        print("🎯 針對 app_enhanced.py 完全優化")
        print("="*70)
        
        try:
            # 1. 設置數據庫
            self.setup_database()
            
            # 2. 載入並預處理數據
            texts, labels = self.load_and_preprocess_data()
            
            # 3. 特徵工程
            X_train, X_test, y_train, y_test = self.prepare_features(texts, labels)
            
            # 4. 初始化模型
            self.initialize_models()
            
            # 5. 訓練和評估模型
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
            
            if not results:
                print("❌ 沒有成功訓練的模型")
                return False
            
            # 6. 選擇最佳模型
            best_model_name, _ = self.select_best_model(results)
            
            if not best_model_name:
                print("❌ 未能選擇最佳模型")
                return False
            
            # 7. 保存性能到數據庫
            self.save_model_performance_to_db(results)
            
            # 8. 創建視覺化
            self.create_visualizations(results, y_test)
            
            # 9. 保存模型文件
            if not self.save_models_for_app(best_model_name):
                return False
            
            # 10. 測試相容性
            if not self.test_app_compatibility():
                return False
            
            print("\n" + "="*70)
            print("✅ 訓練管道執行成功！")
            print(f"🏆 最佳模型: {best_model_name}")
            print("📁 生成文件:")
            print("  ✓ best_model.pkl (主要模型)")
            print("  ✓ model.pkl (相容性)")
            print("  ✓ vectorizer.pkl (向量化器)")
            print("  ✓ all_models.pkl (A/B測試用)")
            print("  ✓ model_performance_report.json (性能報告)")
            print("  ✓ static/*.png (視覺化圖表)")
            print("  ✓ sentiment_analysis.db (數據庫)")
            print("\n🎉 現在可以執行 app_enhanced.py 啟動生產級應用！")
            print("="*70)
            
            return True
            
        except Exception as e:
            print(f"❌ 訓練管道執行失敗: {e}")
            return False

def main():
    """主函數"""
    pipeline = EnhancedSentimentModelPipeline()
    
    print("歡迎使用增強版情感分析模型訓練系統")
    print("此版本完全相容 app_enhanced.py")
    print("-" * 70)
    
    success = pipeline.run_full_pipeline()
    
    if success:
        print("\n🎊 訓練完成！接下來可以:")
        print("1. 執行: python app_enhanced.py")
        print("2. 或使用: ./run.sh")
        print("3. 訪問: http://localhost:5000")
        
        # 提供快速啟動選項
        user_input = input("\n是否立即啟動應用？(y/n): ")
        if user_input.lower() in ['y', 'yes']:
            try:
                os.system("python app_enhanced.py")
            except KeyboardInterrupt:
                print("\n應用已停止")
    else:
        print("\n❌ 訓練失敗，請檢查錯誤信息")

if __name__ == "__main__":
    main()