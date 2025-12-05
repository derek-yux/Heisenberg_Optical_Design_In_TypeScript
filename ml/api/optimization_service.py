from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.layout_optimizer import LayoutOptimizerModel, DesignGoal, OpticalLayout

app = Flask(__name__)
CORS(app)

# Initialize model
model = LayoutOptimizerModel()

# Try to load pre-trained model
try:
    model.load_model('models/trained_optimizer.pkl')
    print("Loaded pre-trained model")
except:
    print("No pre-trained model found, starting fresh")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': len(model.trained_patterns) > 0})

@app.route('/api/optimize', methods=['POST'])
def optimize_layout():
    """Generate optimized layout based on design goal"""
    try:
        data = request.json
        
        # Parse design goal
        goal = DesignGoal(
            goal_type=data.get('goal_type', 'bb84_key_distribution'),
            target_efficiency=data.get('target_efficiency', 0.9),
            max_components=data.get('max_components', 10),
            preferred_components=data.get('preferred_components', [])
        )
        
        # Generate optimized layout
        optimized_layout = model.generate_optimized_layout(goal)
        
        return jsonify({
            'success': True,
            'layout': {
                'components': optimized_layout.components,
                'performance_score': optimized_layout.performance_score,
                'detection_efficiency': optimized_layout.detection_efficiency,
                'photon_loss': optimized_layout.photon_loss
            },
            'metadata': {
                'goal_type': goal.goal_type,
                'model_confidence': 0.85
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model on historical design data"""
    try:
        data = request.json
        design_history = data.get('design_history', [])
        
        if not design_history:
            return jsonify({'success': False, 'error': 'No training data provided'}), 400
        
        model.train_from_history(design_history)
        
        # Save trained model
        model.save_model('models/trained_optimizer.pkl')
        
        return jsonify({
            'success': True,
            'trained_on': len(design_history),
            'best_score': model.trained_patterns[0]['score'] if model.trained_patterns else 0
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/evaluate', methods=['POST'])
def evaluate_layout():
    """Evaluate performance of a given layout"""
    try:
        data = request.json
        layout = data.get('layout', {})
        simulation_results = data.get('simulation_results', {})
        
        features = model.extract_features(layout)
        score = model.calculate_performance_score(layout, simulation_results)
        
        return jsonify({
            'success': True,
            'performance_score': score,
            'features': features.tolist(),
            'recommendations': _generate_recommendations(layout, score)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def _generate_recommendations(layout: Dict, score: float) -> List[str]:
    """Generate improvement recommendations"""
    recommendations = []
    
    if score < 50:
        recommendations.append("Consider adding more detectors to improve detection rate")
        recommendations.append("Optimize component placement to reduce photon loss")
    
    components = layout.get('components', [])
    detector_count = sum(1 for c in components if c['type'] == 'detector')
    
    if detector_count == 0:
        recommendations.append("Add at least one detector to measure photons")
    
    laser_count = sum(1 for c in components if c['type'] == 'laser')
    if laser_count == 0:
        recommendations.append("Add a laser source to emit photons")
    
    if len(components) > 15:
        recommendations.append("Consider simplifying the design - fewer components often perform better")
    
    return recommendations

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)