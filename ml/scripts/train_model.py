import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.layout_optimizer import LayoutOptimizerModel
training_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training_data.json')
try:
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    print(f"Loaded {len(training_data)} training examples from training_data.json")
except FileNotFoundError:
    print(f"Warning: training_data.json not found at {training_data_path}, using fallback data")
    training_data = []


def main():
    print("Training Layout Optimizer Model...")
    
    model = LayoutOptimizerModel()
    model.train_from_history(training_data)
    model_path = '/app/models/trained_optimizer.pkl'
    model.save_model(model_path)
    
    print("\n✓ Model trained and saved successfully!")
    print(f"  - Patterns learned: {len(model.trained_patterns)}")
    print(f"  - Best performance score: {max(model.performance_history):.2f}")

if __name__ == '__main__':
    main()