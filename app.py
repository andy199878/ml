from flask import Flask, request, render_template, jsonify, redirect, url_for
import pickle
import sqlite3
import json
import os
import time
import logging
import random
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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
                app.logger.info(f"✓ 載入 {len(self.models)} 個模型用於A/B測試")
            
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
            
            # 獲取最近N天的統計
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
                model_info = f"模型: {result['model_name']} | 延遲: {result['latency']*1000:.1f}ms"
                
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

@app.route("/api/predict", methods=["POST"])
@timing_decorator
def api_predict():
    """API預測接口"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '請提供文本內容'}), 400
        
        text = data['text'].strip()
        model_name = data.get('model', None)
        
        if not text:
            return jsonify({'error': '文本不能為空'}), 400
        
        # 預測
        result = model_manager.predict_with_timing(text, model_name)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'sentiment': 'positive' if result['prediction'] == 1 else 'negative',
            'model_name': result['model_name'],
            'latency': result['latency'],
            'probabilities': {
                'positive': result['positive_prob'],
                'negative': result['negative_prob']
            }
        })
        
    except Exception as e:
        app.logger.error(f"API預測錯誤: {e}")
        return jsonify({'error': str(e)}), 500

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
    """監控儀表板"""
    try:
        # 獲取模型統計
        stats = model_manager.get_model_stats()
        
        # 獲取預測趨勢數據
        trend_data = get_prediction_trends()
        
        # 生成圖表
        generate_dashboard_charts(stats, trend_data)
        
        return render_template("dashboard.html", 
                             stats=stats,
                             trend_data=trend_data)
                             
    except Exception as e:
        app.logger.error(f"儀表板錯誤: {e}")
        return render_template("dashboard.html", error=str(e))

@app.route("/model_comparison")
@timing_decorator
def model_comparison():
    """模型比較頁面"""
    try:
        # 讀取性能報告
        if os.path.exists("model_performance_report.json"):
            with open("model_performance_report.json", "r") as f:
                performance_data = json.load(f)
        else:
            performance_data = {}
        
        return render_template("model_comparison.html", 
                             performance_data=performance_data)
                             
    except Exception as e:
        app.logger.error(f"模型比較錯誤: {e}")
        return render_template("model_comparison.html", error=str(e))

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """用戶反饋接口"""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        feedback = data.get('feedback')  # 1: correct, 0: incorrect
        
        if prediction_id is None or feedback is None:
            return jsonify({'error': '缺少必要參數'}), 400
        
        # 保存反饋到數據庫
        save_user_feedback(prediction_id, feedback)
        
        return jsonify({'success': True, 'message': '反饋已保存'})
        
    except Exception as e:
        app.logger.error(f"反饋保存錯誤: {e}")
        return jsonify({'error': str(e)}), 500

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

def save_user_feedback(prediction_id, feedback):
    """保存用戶反饋"""
    try:
        conn = sqlite3.connect(model_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET user_feedback = ? 
            WHERE id = ?
        ''', (feedback, prediction_id))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        app.logger.error(f"保存用戶反饋失敗: {e}")

def get_prediction_trends(days=30):
    """獲取預測趨勢數據"""
    try:
        conn = sqlite3.connect(model_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_predictions,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as positive_count
            FROM predictions 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        '''.format(days))
        
        trend_data = []
        for row in cursor.fetchall():
            date, total, avg_conf, positive = row
            trend_data.append({
                'date': date,
                'total_predictions': total,
                'avg_confidence': round(avg_conf, 3) if avg_conf else 0,
                'positive_rate': round(positive / total, 3) if total > 0 else 0
            })
        
        conn.close()
        return trend_data
        
    except Exception as e:
        app.logger.error(f"獲取趨勢數據失敗: {e}")
        return []

def generate_dashboard_charts(stats, trend_data):
    """生成儀表板圖表"""
    try:
        # 1. 模型性能比較圖
        if stats:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('模型實時性能監控', fontsize=16)
            
            models = list(stats.keys())
            total_preds = [stats[m]['total_predictions'] for m in models]
            avg_confs = [stats[m]['avg_confidence'] for m in models]
            avg_lats = [stats[m]['avg_latency'] for m in models]
            pos_rates = [stats[m]['positive_rate'] for m in models]
            
            # 預測數量
            ax1.bar(models, total_preds, color='skyblue')
            ax1.set_title('預測數量')
            ax1.set_ylabel('數量')
            ax1.tick_params(axis='x', rotation=45)
            
            # 平均信心度
            ax2.bar(models, avg_confs, color='lightgreen')
            ax2.set_title('平均信心度')
            ax2.set_ylabel('信心度')
            ax2.tick_params(axis='x', rotation=45)
            
            # 平均延遲
            ax3.bar(models, [lat*1000 for lat in avg_lats], color='orange')
            ax3.set_title('平均延遲 (ms)')
            ax3.set_ylabel('延遲 (ms)')
            ax3.tick_params(axis='x', rotation=45)
            
            # 正面率
            ax4.bar(models, pos_rates, color='pink')
            ax4.set_title('正面預測率')
            ax4.set_ylabel('比例')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('static/dashboard_charts.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 趨勢圖
        if trend_data:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            dates = [item['date'] for item in trend_data]
            totals = [item['total_predictions'] for item in trend_data]
            pos_rates = [item['positive_rate'] for item in trend_data]
            
            # 每日預測數量趨勢
            ax1.plot(dates, totals, marker='o', linewidth=2)
            ax1.set_title('每日預測數量趨勢')
            ax1.set_ylabel('預測數量')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 正面情感比例趨勢
            ax2.plot(dates, pos_rates, marker='s', linewidth=2, color='green')
            ax2.set_title('正面情感比例趨勢')
            ax2.set_ylabel('正面比例')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('static/trend_charts.png', dpi=300, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        app.logger.error(f"生成圖表失敗: {e}")

@app.route("/health")
def health_check():
    """健康檢查接口"""
    try:
        # 檢查模型狀態
        model_status = "OK" if model_manager.best_model is not None else "ERROR"
        vectorizer_status = "OK" if model_manager.vectorizer is not None else "ERROR"
        
        # 檢查數據庫連接
        try:
            conn = sqlite3.connect(model_manager.db_path)
            conn.close()
            db_status = "OK"
        except:
            db_status = "ERROR"
        
        overall_status = "OK" if all(s == "OK" for s in [model_status, vectorizer_status, db_status]) else "ERROR"
        
        return jsonify({
            'status': overall_status,
            'components': {
                'model': model_status,
                'vectorizer': vectorizer_status,
                'database': db_status
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

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
    app.logger.info("- A/B 模型測試")
    app.logger.info("- 實時性能監控")
    app.logger.info("- API 接口")
    app.logger.info("- 健康檢查")
    
    app.run(debug=True, host='0.0.0.0', port=5000)