"""
Fuzzy-Genetic Pricing Intelligence Optimizer
Main Flask Application
Author: Advanced Soft Computing Project
"""

import os
import json
import threading
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

# Import custom modules
from models.data_preprocessing import DataPreprocessor
from models.fuzzy_system import FuzzyPricingSystem
from models.genetic_algorithm import GeneticOptimizer
from models.price_optimizer import PriceOptimizationPipeline
from models.chatgpt_chatbot import ChatGPTPricingAssistant

app = Flask(__name__)
app.secret_key = 'fuzzy-genetic-pricing-optimizer-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize chatbot
chatbot = ChatGPTPricingAssistant()

# Store optimization tasks
optimization_tasks = {}

class OptimizationTask:
    """Background task for optimization"""
    def __init__(self, task_id):
        self.task_id = task_id
        self.status = 'initializing'
        self.progress = 0
        self.result = None
        self.error = None
        self.start_time = datetime.now()
        
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'elapsed_time': str(datetime.now() - self.start_time)
        }

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def run_optimization(task_id, filepath, params):
    """Run optimization in background thread"""
    task = optimization_tasks[task_id]
    
    try:
        # Step 1: Data Preprocessing
        task.status = 'preprocessing'
        task.progress = 10
        
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data(filepath)
        processed_data = preprocessor.preprocess(data)
        
        # Step 2: Feature Engineering
        task.progress = 20
        engineered_data = preprocessor.engineer_features(processed_data)
        
        # Step 3: Fuzzy Logic Analysis
        task.status = 'fuzzy_analysis'
        task.progress = 40
        
        fuzzy_system = FuzzyPricingSystem()
        fuzzy_results = fuzzy_system.analyze_pricing(engineered_data)
        
        # Step 4: Genetic Algorithm Optimization
        task.status = 'genetic_optimization'
        task.progress = 60
        
        optimizer = GeneticOptimizer(
            population_size=params.get('population_size', 100),
            generations=params.get('generations', 100),
            mutation_rate=params.get('mutation_rate', 0.1),
            crossover_rate=params.get('crossover_rate', 0.8)
        )
        
        ga_results = optimizer.optimize(engineered_data, fuzzy_results)
        
        # Step 5: Integration & Final Optimization
        task.status = 'final_optimization'
        task.progress = 80
        
        pipeline = PriceOptimizationPipeline()
        final_results = pipeline.optimize(
            engineered_data, 
            fuzzy_results, 
            ga_results
        )
        
        # Step 6: Generate Recommendations
        task.status = 'generating_recommendations'
        task.progress = 95
        
        recommendations = pipeline.generate_recommendations(final_results)
        summary_stats = pipeline.get_summary_stats(final_results)
        
        # Save results
        result_file = os.path.join(
            app.config['RESULTS_FOLDER'], 
            f'optimization_{task_id}.json'
        )
        
        with open(result_file, 'w') as f:
            json.dump({
                'fuzzy_results': fuzzy_results,
                'ga_results': ga_results,
                'final_results': final_results,
                'recommendations': recommendations,
                'summary_stats': summary_stats
            }, f, default=str)
        
        task.status = 'completed'
        task.progress = 100
        task.result = {
            'file': result_file,
            'summary': summary_stats,
            'recommendations_count': len(recommendations)
        }
        
    except Exception as e:
        task.status = 'failed'
        task.error = str(e)

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/upload')
def upload():
    """Upload page"""
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store filepath in session
        session['uploaded_file'] = filepath
        
        # Preview data
        try:
            preprocessor = DataPreprocessor()
            data = preprocessor.load_data(filepath)
            preview = preprocessor.get_preview(data)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'preview': preview,
                'columns': list(data.columns),
                'shape': data.shape
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/optimize', methods=['POST'])
def start_optimization():
    """Start optimization process"""
    data = request.json
    filepath = session.get('uploaded_file')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Create optimization task
    task_id = str(uuid.uuid4())
    task = OptimizationTask(task_id)
    optimization_tasks[task_id] = task
    
    # Start background thread
    params = {
        'population_size': data.get('population_size', 100),
        'generations': data.get('generations', 100),
        'mutation_rate': data.get('mutation_rate', 0.1),
        'crossover_rate': data.get('crossover_rate', 0.8)
    }
    
    thread = threading.Thread(
        target=run_optimization,
        args=(task_id, filepath, params)
    )
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/api/status/<task_id>')
def get_status(task_id):
    """Get optimization task status"""
    task = optimization_tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(task.to_dict())

@app.route('/dashboard')
def dashboard():
    """Results dashboard"""
    return render_template('dashboard.html')

@app.route('/api/results/<task_id>')
def get_results(task_id):
    """Get optimization results"""
    task = optimization_tasks.get(task_id)
    if not task or task.status != 'completed':
        return jsonify({'error': 'Results not available'}), 404
    
    with open(task.result['file'], 'r') as f:
        results = json.load(f)
    
    return jsonify(results)

@app.route('/results')
def results():
    """Detailed results page"""
    return render_template('results.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/export/<task_id>')
def export_results(task_id):
    """Export results as CSV"""
    task = optimization_tasks.get(task_id)
    if not task or task.status != 'completed':
        return jsonify({'error': 'Results not available'}), 404
    
    with open(task.result['file'], 'r') as f:
        results = json.load(f)
    
    # Create DataFrame from recommendations
    df = pd.DataFrame(results['recommendations'])
    export_path = os.path.join(app.config['RESULTS_FOLDER'], f'export_{task_id}.csv')
    df.to_csv(export_path, index=False)
    
    return send_file(export_path, as_attachment=True, download_name='pricing_recommendations.csv')

@app.route('/api/chat', methods=['POST'])
def chat_with_gpt():
    """Handle chat messages with ChatGPT integration"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        task_id = data.get('task_id')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get product context if available
        product_context = None
        optimization_results = None
        
        if task_id and task_id in optimization_tasks:
            task = optimization_tasks[task_id]
            if task.status == 'completed':
                with open(task.result['file'], 'r') as f:
                    results = json.load(f)
                
                optimization_results = results
                
                # Check if user is asking about a specific product
                for rec in results.get('recommendations', []):
                    if rec['product_name'].lower() in user_message.lower():
                        # Get full product data
                        product_data = next(
                            (p for p in results['final_results']['optimized_prices'] 
                             if p['product_id'] == rec['product_id']), None
                        )
                        if product_data:
                            product_context = {
                                **rec,
                                **product_data,
                                'competitor_price': product_data.get('competitor_price')
                            }
                        break
        
        # Get response from ChatGPT
        response = chatbot.chat(user_message, product_context, optimization_results)
        
        # Get suggested follow-up questions
        product_name = product_context['product_name'] if product_context else None
        suggestions = chatbot.get_suggested_questions(product_name)
        
        return jsonify({
            'response': response.get('response', 'Sorry, I encountered an error.'),
            'suggestions': suggestions[:4],
            'model': response.get('model', 'fallback'),
            'usage': response.get('usage', {}),
            'fallback': response.get('fallback', False)
        })
        
    except Exception as e:
        return jsonify({
            'response': "I'm having trouble connecting to my brain right now. Please try again in a moment.",
            'error': str(e),
            'suggestions': chatbot.get_suggested_questions()
        }), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat_history():
    """Clear conversation history"""
    result = chatbot.clear_history()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)