from flask import Flask, request, render_template, jsonify
import pickle
import sqlite3
import json
import os
import time
import logging
import random
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 生產環境中請使用環境變量

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# 新增：初始化資料庫（與 train_model.py 結構一致）
def init_database(db_path: str = 'sentiment_analysis.db') -> None:
    try:
        conn = sqlite3.connect(db_path)
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
        app.logger.info('✓ 資料庫已初始化')
    except Exception as e:
        app.logger.error(f'初始化資料庫失敗: {e}')

class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.best_model = None
        self.db_path = 'sentiment_analysis.db'
        self.load_models()
    
    def load_models(self):
        """載入所有可用模型"""
        try:
            # 載入最佳模型
            if os.path.exists("best_model.pkl"):
                self.best_model = pickle.load(open("best_model.pkl", "rb"))
                app.logger.info("✓ 最佳模型載入成功")
            elif os.path.exists("model.pkl"):
                self.best_model = pickle.load(open("model.pkl", "rb"))
                app.logger.info("✓ 預設模型載入成功")
            
            # 載入向量化器
            if os.path.exists("vectorizer.pkl"):
                self.vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
                app.logger.info("✓ 向量化器載入成功")
            
            # 載入所有模型（用於A/B測試）
            if os.path.exists("all_models.pkl"):
                self.models = pickle.load(open("all_models.pkl", "rb"))
                app.logger.info(f"✓ 載入 {len(self.models)} 個模型用於A/B測試 (已過濾XGBoost)")
            
        except Exception as e:
            app.logger.error(f"模型載入失敗: {e}")
    
    def predict_with_timing(self, text, model_name=None):
        """帶計時的預測"""
        start_time = time.time()
        
        try:
            if self.vectorizer is None:
                raise ValueError("向量化器未載入")
            
            # 選擇模型
            if model_name and model_name in self.models:
                model = self.models[model_name]
            elif self.best_model:
                model = self.best_model
                if os.path.exists("model_performance_report.json"):
                    with open("model_performance_report.json", "r") as f:
                        performance_data = json.load(f)
                        model_name = performance_data['best_model']
                else:
                    model_name = "best_model"
            else:
                raise ValueError("沒有可用的模型")
            
            # 向量化文本
            vec = self.vectorizer.transform([text])
            
            # 預測
            prediction = model.predict(vec)[0]
            probability = model.predict_proba(vec)[0]
            confidence = max(probability)
            
            # 計算延遲
            latency = time.time() - start_time
            
            # 記錄到數據庫
            self.save_prediction(text, prediction, confidence, model_name, latency)

            
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'model_name': model_name,
                'latency': latency,
                'positive_prob': float(probability[1]),
                'negative_prob': float(probability[0])
            }
            
        except Exception as e:
            app.logger.error(f"預測失敗: {e}")
            raise
    
    def save_prediction(self, text, prediction, confidence, model_name, latency):
        """保存預測結果到數據庫"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (text, prediction, confidence, model_name, latency)
                VALUES (?, ?, ?, ?, ?)
            ''', (text, int(prediction), float(confidence), model_name, float(latency)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            app.logger.error(f"保存預測失敗: {e}")
    
    def get_model_stats(self, days=7):
        """獲取模型統計數據"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    model_name,
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(latency) as avg_latency,
                    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as negative_count
                FROM predictions 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY model_name
            '''.format(days))
            
            stats = {}
            for row in cursor.fetchall():
                model_name, total, avg_conf, avg_lat, pos, neg = row
                # 過濾掉 XGBoost 與 best_model
                name_lower = model_name.lower()
                if 'xgb' in name_lower or 'xgboost' in name_lower or model_name == 'best_model':
                    continue
                stats[model_name] = {
                    'total_predictions': total,
                    'avg_confidence': round(avg_conf, 3) if avg_conf else 0,
                    'avg_latency': round(avg_lat, 4) if avg_lat else 0,
                    'positive_rate': round(pos / total, 3) if total > 0 else 0,
                    'negative_rate': round(neg / total, 3) if total > 0 else 0
                }
            
            conn.close()
            return stats
            
        except Exception as e:
            app.logger.error(f"獲取統計失敗: {e}")
            return {}

# 啟動時初始化資料庫，確保表存在（WSGI/本地皆適用）
init_database('sentiment_analysis.db')

# 創建模型管理器實例
model_manager = ModelManager()

def timing_decorator(f):
    """計時裝飾器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        app.logger.info(f"{f.__name__} 執行時間: {end_time - start_time:.4f}秒")
        return result
    return wrapper

@app.route("/", methods=["GET", "POST"])
@timing_decorator
def index():
    """主頁面 - 情感分析"""
    prediction = ""
    confidence = ""
    error_message = ""
    model_info = ""
    
    if request.method == "POST":
        text = request.form.get("review", "").strip()
        
        if not text:
            error_message = "請輸入要分析的文字"
        else:
            try:
                # 預測
                result = model_manager.predict_with_timing(text)
                
                # 格式化結果
                if result['prediction'] == 1:
                    prediction = "Positive 😀"
                else:
                    prediction = "Negative 😡"
                
                confidence = f"信心度: {result['confidence']:.1%}"
                # 移除模型名稱輸出，只顯示延遲
                model_info = f"延遲: {result['latency']*1000:.1f}ms"
                
                # 如果信心度較低，提供額外信息
                if result['confidence'] < 0.7:
                    confidence += " (較不確定)"
                
            except Exception as e:
                error_message = f"預測時發生錯誤: {str(e)}"
                app.logger.error(f"預測錯誤: {e}")
    
    return render_template("enhanced_index.html", 
                         prediction=prediction, 
                         confidence=confidence,
                         error_message=error_message,
                         model_info=model_info)


@app.route("/ab_test", methods=["GET", "POST"])
@timing_decorator
def ab_test():
    """A/B測試頁面"""
    if request.method == "POST":
        text = request.form.get("review", "").strip()
        
        if text and len(model_manager.models) >= 2:
            try:
                # 隨機選擇兩個不同的模型
                model_names = list(model_manager.models.keys())
                model_a, model_b = random.sample(model_names, 2)
                
                # 分別預測
                result_a = model_manager.predict_with_timing(text, model_a)
                result_b = model_manager.predict_with_timing(text, model_b)
                
                # 保存A/B測試結果
                save_ab_test_result(model_a, model_b, text, result_a, result_b)
                
                return render_template("ab_test.html",
                                     text=text,
                                     model_a=model_a,
                                     model_b=model_b,
                                     result_a=result_a,
                                     result_b=result_b)
                                     
            except Exception as e:
                app.logger.error(f"A/B測試錯誤: {e}")
                return render_template("ab_test.html", error=str(e))
    
    return render_template("ab_test.html")

@app.route("/dashboard")
@timing_decorator
def dashboard():
    """監控儀表板（簡化版）"""
    try:
        # 獲取模型統計
        stats = model_manager.get_model_stats()
        
        # 聚合統計
        aggregates = None
        if stats:
            values = list(stats.values())
            total_predictions = sum(v['total_predictions'] for v in values)
            avg_confidence = (sum(v['avg_confidence'] for v in values) / len(values)) if values else 0
            avg_latency_ms = (sum(v['avg_latency'] for v in values) / len(values) * 1000) if values else 0
            positive_rate = (sum(v['positive_rate'] for v in values) / len(values)) if values else 0
            aggregates = {
                'total_predictions': total_predictions,
                'avg_confidence': round(avg_confidence, 3),
                'avg_latency_ms': round(avg_latency_ms, 1),
                'positive_rate': round(positive_rate, 3)
            }
        
        return render_template("dashboard.html", 
                             stats=stats,
                             aggregates=aggregates)
                             
    except Exception as e:
        app.logger.error(f"儀表板錯誤: {e}")
        return render_template("dashboard.html", error=str(e))

@app.route("/model_comparison")
@timing_decorator
def model_comparison():
    """模型比較頁面（簡化版）"""
    try:
        if os.path.exists("model_performance_report.json"):
            with open("model_performance_report.json", "r") as f:
                performance_data = json.load(f)
            # 過濾輸出端的 best_model 與 XGBoost 條目
            if isinstance(performance_data, dict):
                if 'performances' in performance_data and isinstance(performance_data['performances'], dict):
                    performance_data['performances'] = {
                        k: v for k, v in performance_data['performances'].items()
                    }
        else:
            performance_data = {}
        
        return render_template("model_comparison.html", 
                             performance_data=performance_data)
                             
    except Exception as e:
        app.logger.error(f"模型比較錯誤: {e}")
        return render_template("model_comparison.html", error=str(e))

def save_ab_test_result(model_a, model_b, text, result_a, result_b):
    """保存A/B測試結果"""
    try:
        conn = sqlite3.connect(model_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ab_test_results 
            (model_a, model_b, text, prediction_a, prediction_b, 
             confidence_a, confidence_b)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_a, model_b, text,
            result_a['prediction'], result_b['prediction'],
            result_a['confidence'], result_b['confidence']
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        app.logger.error(f"保存A/B測試結果失敗: {e}")


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"內部錯誤: {error}")
    return render_template('500.html'), 500

if __name__ == "__main__":
    # 確保靜態文件夾存在
    os.makedirs('static', exist_ok=True)
    
    app.logger.info("啟動增強版情感分析應用...")
    app.logger.info("功能包括:")
    app.logger.info("- 基本情感分析")
    app.logger.info("- A/B 測試")
    app.logger.info("- 儀表板（簡化統計）")
    app.logger.info("- API 接口")
    app.logger.info("- 健康檢查")
    
    app.run(debug=True, host='0.0.0.0', port=5000)