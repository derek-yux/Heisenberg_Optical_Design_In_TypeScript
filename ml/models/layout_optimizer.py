import numpy as np
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pickle

@dataclass
class DesignGoal:
    """Represents design objectives"""
    goal_type: str  # 'maximize_detection', 'minimize_loss', 'bb84_key_distribution'
    target_efficiency: float = 0.9
    max_components: int = 10
    preferred_components: List[str] = None

@dataclass
class OpticalLayout:
    """Represents a complete optical system layout"""
    components: List[Dict]
    performance_score: float
    detection_efficiency: float
    photon_loss: float
    
class LayoutOptimizerModel:
    """Neural network-inspired optimizer for optical layouts"""
    
    def __init__(self):
        self.component_types = ['laser', 'mirror', 'beamsplitter', 'polarizer', 'detector', 'waveplate']
        self.trained_patterns = []
        self.performance_history = []
        
    def extract_features(self, layout: Dict) -> np.ndarray:
        """Extract features from optical layout for ML model"""
        features = []
        
        # Component count features
        component_counts = {ct: 0 for ct in self.component_types}
        for comp in layout.get('components', []):
            comp_type = comp.get('type')
            if comp_type in component_counts:
                component_counts[comp_type] += 1
        
        features.extend([component_counts[ct] for ct in self.component_types])
        
        # Spatial distribution features
        if layout.get('components'):
            positions = [(c['x'], c['y']) for c in layout['components']]
            avg_x = np.mean([p[0] for p in positions])
            avg_y = np.mean([p[1] for p in positions])
            std_x = np.std([p[0] for p in positions])
            std_y = np.std([p[1] for p in positions])
            features.extend([avg_x, avg_y, std_x, std_y])
        else:
            features.extend([0, 0, 0, 0])
        
        # Connectivity features (simplified)
        num_components = len(layout.get('components', []))
        features.append(num_components)
        
        # Path length estimate
        total_distance = 0
        if len(layout.get('components', [])) > 1:
            for i in range(len(layout['components']) - 1):
                c1 = layout['components'][i]
                c2 = layout['components'][i + 1]
                dist = np.sqrt((c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2)
                total_distance += dist
        features.append(total_distance)
        
        return np.array(features)
    
    def calculate_performance_score(self, layout: Dict, simulation_results: Dict) -> float:
        """Calculate performance score from simulation results"""
        score = 0.0
        
        # Detection efficiency
        detections = simulation_results.get('detections', 0)
        emissions = simulation_results.get('emissions', 1)
        detection_rate = detections / emissions if emissions > 0 else 0
        score += detection_rate * 40
        
        # Photon loss penalty
        loss_events = simulation_results.get('loss_events', 0)
        loss_penalty = loss_events * 5
        score -= loss_penalty
        
        # Component efficiency
        avg_intensity = simulation_results.get('avg_intensity', 0)
        score += avg_intensity * 30
        
        # Complexity penalty (prefer simpler designs)
        num_components = len(layout.get('components', []))
        complexity_penalty = (num_components - 3) * 2 if num_components > 3 else 0
        score -= complexity_penalty
        
        # BB84 specific bonus
        if simulation_results.get('bb84_compatible', False):
            score += 20
        
        return max(0, min(100, score))
    
    def train_from_history(self, design_history: List[Dict]):
        """Train model from historical designs and their performance"""
        print(f"Training on {len(design_history)} historical designs...")
        
        for entry in design_history:
            layout = entry['layout']
            results = entry['simulation_results']
            
            features = self.extract_features(layout)
            score = self.calculate_performance_score(layout, results)
            
            self.trained_patterns.append({
                'features': features,
                'score': score,
                'layout': layout
            })
            self.performance_history.append(score)
        
        # Sort by performance
        self.trained_patterns.sort(key=lambda x: x['score'], reverse=True)
        print(f"Training complete. Best score: {self.trained_patterns[0]['score']:.2f}")
    
    def generate_optimized_layout(self, goal: DesignGoal) -> OpticalLayout:
        """Generate optimized layout based on design goal using learned patterns"""
        
        # Start with best performing pattern as template
        if self.trained_patterns:
            base_layout = self.trained_patterns[0]['layout'].copy()
        else:
            # Default BB84 layout if no training data
            base_layout = self._get_default_bb84_layout()
        
        # Optimize based on goal
        if goal.goal_type == 'maximize_detection':
            optimized = self._optimize_for_detection(base_layout, goal)
        elif goal.goal_type == 'minimize_loss':
            optimized = self._optimize_for_low_loss(base_layout, goal)
        elif goal.goal_type == 'bb84_key_distribution':
            optimized = self._optimize_for_bb84(base_layout, goal)
        else:
            optimized = base_layout
        
        # Calculate expected performance
        expected_efficiency = self._estimate_efficiency(optimized)
        expected_loss = self._estimate_loss(optimized)
        expected_score = 85.0  # Placeholder
        
        return OpticalLayout(
            components=optimized['components'],
            performance_score=expected_score,
            detection_efficiency=expected_efficiency,
            photon_loss=expected_loss
        )
    
    def _optimize_for_detection(self, base_layout: Dict, goal: DesignGoal) -> Dict:
        """Optimize layout to maximize detection efficiency"""
        layout = base_layout.copy()
        components = layout['components'].copy()
        
        # Ensure we have detectors
        detector_count = sum(1 for c in components if c['type'] == 'detector')
        if detector_count == 0:
            components.append({
                'id': f'detector_{len(components)}',
                'type': 'detector',
                'x': 600,
                'y': 200,
                'rotation': 0,
                'properties': {'efficiency': 0.98, 'label': 'Primary Detector'}
            })
        
        # Optimize detector efficiency
        for comp in components:
            if comp['type'] == 'detector':
                comp['properties']['efficiency'] = min(0.99, goal.target_efficiency)
        
        layout['components'] = components
        return layout
    
    def _optimize_for_low_loss(self, base_layout: Dict, goal: DesignGoal) -> Dict:
        """Optimize layout to minimize photon loss"""
        layout = base_layout.copy()
        components = layout['components'].copy()
        
        # Increase component reflectivity/efficiency
        for comp in components:
            if comp['type'] == 'mirror' and 'reflectivity' in comp.get('properties', {}):
                comp['properties']['reflectivity'] = 0.99
            elif comp['type'] == 'beamsplitter' and 'reflectivity' in comp.get('properties', {}):
                # Optimize for transmission
                comp['properties']['reflectivity'] = 0.3
        
        # Remove unnecessary components
        essential_types = {'laser', 'detector'}
        if len(components) > goal.max_components:
            components = [c for c in components if c['type'] in essential_types or 
                         components.index(c) < goal.max_components]
        
        layout['components'] = components
        return layout
    
    def _optimize_for_bb84(self, base_layout: Dict, goal: DesignGoal) -> Dict:
        """Optimize for BB84 quantum key distribution"""
        # Return optimized BB84 configuration
        components = [
            {
                'id': 'laser_alice',
                'type': 'laser',
                'x': 100,
                'y': 200,
                'rotation': 0,
                'properties': {'state': 'H', 'angle': 0, 'label': 'Alice'}
            },
            {
                'id': 'polarizer_alice',
                'type': 'polarizer',
                'x': 200,
                'y': 200,
                'rotation': 0,
                'properties': {'basis': 'rectilinear', 'angle': 0, 'label': 'Alice Basis'}
            },
            {
                'id': 'beamsplitter_channel',
                'type': 'beamsplitter',
                'x': 350,
                'y': 200,
                'rotation': 0,
                'properties': {'reflectivity': 0.5, 'label': 'Quantum Channel'}
            },
            {
                'id': 'polarizer_bob',
                'type': 'polarizer',
                'x': 500,
                'y': 200,
                'rotation': 0,
                'properties': {'basis': 'rectilinear', 'angle': 0, 'label': 'Bob Basis'}
            },
            {
                'id': 'detector_bob',
                'type': 'detector',
                'x': 600,
                'y': 200,
                'rotation': 0,
                'properties': {'efficiency': goal.target_efficiency, 'label': 'Bob'}
            }
        ]
        
        return {'components': components}
    
    def _get_default_bb84_layout(self) -> Dict:
        """Return default BB84 layout"""
        goal = DesignGoal(goal_type='bb84_key_distribution', target_efficiency=0.95)
        return self._optimize_for_bb84({}, goal)
    
    def _estimate_efficiency(self, layout: Dict) -> float:
        """Estimate detection efficiency from layout"""
        detector_count = sum(1 for c in layout['components'] if c['type'] == 'detector')
        if detector_count == 0:
            return 0.0
        
        # Simple heuristic
        avg_efficiency = 0.85
        for comp in layout['components']:
            if comp['type'] == 'detector':
                avg_efficiency = max(avg_efficiency, comp['properties'].get('efficiency', 0.85))
        
        return avg_efficiency
    
    def _estimate_loss(self, layout: Dict) -> float:
        """Estimate photon loss from layout"""
        loss = 0.0
        
        for comp in layout['components']:
            if comp['type'] == 'beamsplitter':
                loss += 0.5  # 50% splitting
            elif comp['type'] == 'polarizer':
                loss += 0.2  # Polarization filtering
            elif comp['type'] == 'mirror':
                loss += 0.05  # Small reflection loss
        
        return min(1.0, loss)
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'trained_patterns': self.trained_patterns,
                'performance_history': self.performance_history
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.trained_patterns = data['trained_patterns']
            self.performance_history = data['performance_history']
        print(f"Model loaded from {filepath}")