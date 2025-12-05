import sys
sys.path.append('..')

from models.layout_optimizer import LayoutOptimizerModel
import json

# Sample training data
training_data = [
    {
        'layout': {
            'components': [
                {'id': 'l1', 'type': 'laser', 'x': 100, 'y': 200, 'rotation': 0, 'properties': {}},
                {'id': 'd1', 'type': 'detector', 'x': 600, 'y': 200, 'rotation': 0, 'properties': {'efficiency': 0.95}}
            ]
        },
        'simulation_results': {
            'detections': 95,
            'emissions': 100,
            'loss_events': 5,
            'avg_intensity': 0.9,
            'bb84_compatible': False
        }
    },
    {
        'layout': {
            'components': [
                {'id': 'l1', 'type': 'laser', 'x': 100, 'y': 200, 'rotation': 0, 'properties': {}},
                {'id': 'p1', 'type': 'polarizer', 'x': 200, 'y': 200, 'rotation': 0, 'properties': {}},
                {'id': 'bs1', 'type': 'beamsplitter', 'x': 350, 'y': 200, 'rotation': 0, 'properties': {}},
                {'id': 'p2', 'type': 'polarizer', 'x': 500, 'y': 200, 'rotation': 0, 'properties': {}},
                {'id': 'd1', 'type': 'detector', 'x': 600, 'y': 200, 'rotation': 0, 'properties': {'efficiency': 0.92}}
            ]
        },
        'simulation_results': {
            'detections': 85,
            'emissions': 100,
            'loss_events': 15,
            'avg_intensity': 0.75,
            'bb84_compatible': True
        }
    }
]

def main():
    print("Training Layout Optimizer Model...")
    
    model = LayoutOptimizerModel()
    model.train_from_history(training_data)
    
    # Save model
    model.save_model('../models/trained_optimizer.pkl')
    
    print("\n✓ Model trained and saved successfully!")
    print(f"  - Patterns learned: {len(model.trained_patterns)}")
    print(f"  - Best performance score: {max(model.performance_history):.2f}")

if __name__ == '__main__':
    main()