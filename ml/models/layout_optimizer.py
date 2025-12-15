import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

@dataclass
class DesignGoal:
    goal_type: str
    target_efficiency: float = 0.9
    max_components: int = 10
    preferred_components: List[str] = None
    laser_range: Tuple[int, int] = (1, 3)
    mirror_range: Tuple[int, int] = (0, 4)
    beamsplitter_range: Tuple[int, int] = (0, 4)
    polarizer_range: Tuple[int, int] = (0, 4)
    detector_range: Tuple[int, int] = (1, 4)
    waveplate_range: Tuple[int, int] = (0, 3)

@dataclass
class OpticalLayout:
    components: List[Dict]
    performance_score: float
    detection_efficiency: float
    photon_loss: float
    provenance: str = 'ml_optimized'
    explanation: Optional[List[str]] = None
    score_breakdown: Optional[Dict[str, float]] = None
    
class LayoutOptimizerModel:
    def __init__(self):
        self.component_types = ['laser', 'mirror', 'beamsplitter', 'polarizer', 'detector', 'waveplate']
        self.trained_patterns = []
        self.performance_history = []
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, layout: Dict) -> np.ndarray:
        features = []
        component_counts = {ct: 0 for ct in self.component_types}
        for comp in layout.get('components', []):
            comp_type = comp.get('type')
            if comp_type in component_counts:
                component_counts[comp_type] += 1
        
        features.extend([component_counts[ct] for ct in self.component_types])

        total_components = sum(component_counts.values())
        detector_ratio = component_counts['detector'] / total_components if total_components > 0 else 0
        laser_ratio = component_counts['laser'] / total_components if total_components > 0 else 0
        beamsplitter_ratio = component_counts['beamsplitter'] / total_components if total_components > 0 else 0
        features.extend([detector_ratio, laser_ratio, beamsplitter_ratio])

        estimated_transmission = (0.5 ** component_counts['beamsplitter']) * (0.8 ** component_counts['polarizer'])
        estimated_mirror_loss = 1 - (0.95 ** component_counts['mirror'])
        laser_to_detector_ratio = component_counts['laser'] / max(component_counts['detector'], 1)
        features.extend([estimated_transmission, estimated_mirror_loss, laser_to_detector_ratio, total_components])

        if layout.get('components'):
            positions = [(c['x'], c['y']) for c in layout['components']]
            avg_x = np.mean([p[0] for p in positions])
            avg_y = np.mean([p[1] for p in positions])
            std_x = np.std([p[0] for p in positions])
            std_y = np.std([p[1] for p in positions])
            features.extend([avg_x, avg_y, std_x, std_y])
        else:
            features.extend([0, 0, 0, 0])
        
        total_distance = 0
        if len(layout.get('components', [])) > 1:
            for i in range(len(layout['components']) - 1):
                c1 = layout['components'][i]
                c2 = layout['components'][i + 1]
                dist = np.sqrt((c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2)
                total_distance += dist
        
        avg_distance = total_distance / max(len(layout.get('components', [])) - 1, 1)
        features.extend([total_distance, avg_distance])
        
        return np.array(features)
    
    def calculate_performance_score(self, layout: Dict, simulation_results: Dict) -> float:
        score = 0.0
        
        detections = simulation_results.get('detections', 0)
        emissions = simulation_results.get('emissions', 1)
        detection_rate = detections / emissions if emissions > 0 else 0
        score += detection_rate * 50
        
        loss_events = simulation_results.get('loss_events', 0)
        loss_penalty = loss_events * 3
        score -= loss_penalty
        
        avg_intensity = simulation_results.get('avg_intensity', 0)
        score += avg_intensity * 40
        num_components = len(layout.get('components', []))
        complexity_penalty = (num_components - 5) * 1.5 if num_components > 5 else 0
        score -= complexity_penalty
        
        return max(0, min(100, score))
    
    def train_from_history(self, design_history: List[Dict]):
        if len(design_history) == 0:
            print("No training data provided. Model will use untrained defaults.")
            return
            
        print(f"Training ML model on {len(design_history)} historical designs...")
        
        X = []
        y = []
        
        for entry in design_history:
            layout = entry['layout']
            results = entry['simulation_results']
            
            features = self.extract_features(layout)
            score = self.calculate_performance_score(layout, results)
            
            X.append(features)
            y.append(score)
            
            self.trained_patterns.append({
                'features': features,
                'score': score,
                'layout': layout
            })
            self.performance_history.append(score)
        X = np.array(X)
        y = np.array(y)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        self.trained_patterns.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"✓ ML model trained successfully!")
        print(f"  - Training samples: {len(X)}")
        print(f"  - Best score in training data: {max(y):.2f}")
        print(f"  - Average score: {np.mean(y):.2f}")
        print(f"  - Model R² score: {self.model.score(X_scaled, y):.3f}")
    
    def generate_optimized_layout(self, goal: DesignGoal) -> OpticalLayout:
        if not self.is_trained:
            print("⚠️  Model not trained! Using fallback heuristics.")
            return self._generate_fallback_layout(goal)

        num_candidates = 50
        num_pattern_based = int(num_candidates * 0.8)
        num_random = num_candidates - num_pattern_based
        
        print(f"Generating {num_candidates} candidate layouts ({num_pattern_based} pattern-based, {num_random} random)...")
        
        candidates = []

        top_patterns = self.trained_patterns[:5]
        for i in range(num_pattern_based):
            pattern_idx = i % len(top_patterns)
            candidate_components = self._generate_from_pattern(top_patterns[pattern_idx]['layout'], goal, seed=i)
            candidates.append({'components': candidate_components})

        for i in range(num_random):
            candidate_components = self._generate_random_layout(goal, seed=i + 1000)
            candidates.append({'components': candidate_components})

        predictions = []
        for candidate in candidates:
            features = self.extract_features(candidate)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            predicted_score = self.model.predict(features_scaled)[0]
            predictions.append(predicted_score)

        best_idx = np.argmax(predictions)
        best_layout = candidates[best_idx]
        predicted_score = predictions[best_idx]
        
        print(f"✓ Best candidate predicted score: {predicted_score:.2f}")
        print(f"✓ Score range: {min(predictions):.1f} - {max(predictions):.1f}")

        expected_efficiency = self._estimate_efficiency(best_layout)
        expected_loss = self._estimate_loss(best_layout)

        detections = int(expected_efficiency * 100)
        emissions = 100
        loss_events = int(expected_loss * 10)
        avg_intensity = expected_efficiency
        
        simulation_results = {
            'detections': detections,
            'emissions': emissions,
            'loss_events': loss_events,
            'avg_intensity': avg_intensity,
        }
        
        actual_score = self.calculate_performance_score(best_layout, simulation_results)

        detection_component = (detections / emissions) * 50 if emissions > 0 else 0
        loss_penalty = loss_events * 3
        intensity_component = avg_intensity * 40
        num_components = len(best_layout['components'])
        complexity_penalty = (num_components - 5) * 1.5 if num_components > 5 else 0
        
        score_breakdown = {
            'detection_component': detection_component,
            'intensity_component': intensity_component,
            'loss_penalty': -loss_penalty,
            'complexity_penalty': -complexity_penalty,
            'final_score': actual_score
        }

        comp_counts = {}
        for c in best_layout['components']:
            comp_counts[c['type']] = comp_counts.get(c['type'], 0) + 1

        is_pattern_based = best_idx < num_pattern_based

        explanation: List[str] = []
        explanation.append(f"ML model evaluated {num_candidates} candidates")
        if is_pattern_based:
            explanation.append(f"Selected layout based on high-performing training pattern")
        else:
            explanation.append(f"Selected layout from random exploration")
        explanation.append(f"Config: {comp_counts.get('laser', 0)}L + {comp_counts.get('beamsplitter', 0)}BS + {comp_counts.get('mirror', 0)}M + {comp_counts.get('detector', 0)}D")
        explanation.append(f"Predicted score: {predicted_score:.1f}, Actual: {actual_score:.1f}")
        explanation.append(f"Detection efficiency: {expected_efficiency:.1%}")
        explanation.append(f"Photon loss: {expected_loss:.1%}")
        
        return OpticalLayout(
            components=best_layout['components'],
            performance_score=actual_score,
            detection_efficiency=expected_efficiency,
            photon_loss=expected_loss,
            provenance='ml_random_forest',
            explanation=explanation,
            score_breakdown=score_breakdown
        )
    
    def _generate_from_pattern(self, pattern_layout: Dict, goal: DesignGoal, seed: int = 0) -> List[Dict]:
        np.random.seed(seed)
        
        pattern_counts = {}
        for comp in pattern_layout.get('components', []):
            comp_type = comp['type']
            pattern_counts[comp_type] = pattern_counts.get(comp_type, 0) + 1

        laser_count = int(np.clip(pattern_counts.get('laser', 1), goal.laser_range[0], goal.laser_range[1]))
        mirror_count = int(np.clip(pattern_counts.get('mirror', 0), goal.mirror_range[0], goal.mirror_range[1]))
        beamsplitter_count = int(np.clip(pattern_counts.get('beamsplitter', 0), goal.beamsplitter_range[0], goal.beamsplitter_range[1]))
        polarizer_count = int(np.clip(pattern_counts.get('polarizer', 0), goal.polarizer_range[0], goal.polarizer_range[1]))
        detector_count = int(np.clip(pattern_counts.get('detector', 1), goal.detector_range[0], goal.detector_range[1]))
        waveplate_count = int(np.clip(pattern_counts.get('waveplate', 0), goal.waveplate_range[0], goal.waveplate_range[1]))
    
        components: List[Dict] = []
        x_pos = 100
        y_positions = np.linspace(150, 450, max(laser_count, 1))
        
        for i in range(laser_count):
            components.append({
                'id': f'laser_{i}',
                'type': 'laser',
                'x': x_pos,
                'y': float(y_positions[i]) if i < len(y_positions) else 250,
                'rotation': 0,
                'properties': {
                    'state': 'H' if i % 2 == 0 else 'V',
                    'angle': int((i * 45) % 180),
                    'label': f'L{i+1}'
                }
            })
        
        x_pos += 120
        
        for i in range(polarizer_count):
            y_offset = i * 80 - (polarizer_count - 1) * 40
            components.append({
                'id': f'polarizer_{i}',
                'type': 'polarizer',
                'x': x_pos,
                'y': 250 + y_offset,
                'rotation': 0,
                'properties': {
                    'basis': 'rectilinear' if i % 2 == 0 else 'diagonal',
                    'angle': int((i * 45) % 180),
                    'label': f'P{i+1}'
                }
            })
        
        if polarizer_count > 0:
            x_pos += 120
        
        for i in range(waveplate_count):
            y_offset = i * 80 - (waveplate_count - 1) * 40
            components.append({
                'id': f'waveplate_{i}',
                'type': 'waveplate',
                'x': x_pos,
                'y': 250 + y_offset,
                'rotation': 0,
                'properties': {
                    'waveplatetype': 'quarter' if i % 2 == 0 else 'half',
                    'angle': 45 + (i * 22.5),
                    'label': f'W{i+1}'
                }
            })
        
        if waveplate_count > 0:
            x_pos += 120
        
        for i in range(beamsplitter_count):
            y_offset = i * 100 - (beamsplitter_count - 1) * 50
            components.append({
                'id': f'beamsplitter_{i}',
                'type': 'beamsplitter',
                'x': x_pos,
                'y': 250 + y_offset,
                'rotation': 0,
                'properties': {
                    'reflectivity': 0.5,
                    'label': f'BS{i+1}'
                }
            })
        
        if beamsplitter_count > 0:
            x_pos += 120
        
        for i in range(mirror_count):
            y_offset = i * 100 - (mirror_count - 1) * 50
            components.append({
                'id': f'mirror_{i}',
                'type': 'mirror',
                'x': x_pos,
                'y': 250 + y_offset,
                'rotation': 45,
                'properties': {
                    'reflectivity': 0.98,
                    'label': f'M{i+1}'
                }
            })
        
        if mirror_count > 0:
            x_pos += 120
        
        detector_y_positions = np.linspace(150, 450, detector_count)
        for i in range(detector_count):
            components.append({
                'id': f'detector_{i}',
                'type': 'detector',
                'x': x_pos,
                'y': float(detector_y_positions[i]),
                'rotation': 0,
                'properties': {
                    'efficiency': min(0.98, goal.target_efficiency),
                    'label': f'D{i+1}'
                }
            })
        
        return components
    
    def _generate_random_layout(self, goal: DesignGoal, seed: int = 0) -> List[Dict]:
        np.random.seed(seed)
        
        laser_count = int(np.random.randint(goal.laser_range[0], goal.laser_range[1] + 1))
        mirror_count = int(np.random.randint(goal.mirror_range[0], goal.mirror_range[1] + 1))
        beamsplitter_count = int(np.random.randint(goal.beamsplitter_range[0], goal.beamsplitter_range[1] + 1))
        polarizer_count = int(np.random.randint(goal.polarizer_range[0], goal.polarizer_range[1] + 1))
        detector_count = int(np.random.randint(goal.detector_range[0], goal.detector_range[1] + 1))
        waveplate_count = int(np.random.randint(goal.waveplate_range[0], goal.waveplate_range[1] + 1))
        
        components: List[Dict] = []
        x_pos = 100
        y_positions = np.linspace(150, 450, max(laser_count, 1))
        
        for i in range(laser_count):
            components.append({
                'id': f'laser_{i}',
                'type': 'laser',
                'x': x_pos,
                'y': float(y_positions[i]) if i < len(y_positions) else 250,  # Convert numpy float
                'rotation': 0,
                'properties': {
                    'state': 'H' if i % 2 == 0 else 'V',
                    'angle': int((i * 45) % 180),
                    'label': f'L{i+1}'
                }
            })
        
        x_pos += 120
        
        for i in range(polarizer_count):
            y_offset = i * 80 - (polarizer_count - 1) * 40
            components.append({
                'id': f'polarizer_{i}',
                'type': 'polarizer',
                'x': x_pos,
                'y': 250 + y_offset,
                'rotation': 0,
                'properties': {
                    'basis': 'rectilinear' if i % 2 == 0 else 'diagonal',
                    'angle': int((i * 45) % 180),
                    'label': f'P{i+1}'
                }
            })
        
        if polarizer_count > 0:
            x_pos += 120
        
        for i in range(waveplate_count):
            y_offset = i * 80 - (waveplate_count - 1) * 40
            components.append({
                'id': f'waveplate_{i}',
                'type': 'waveplate',
                'x': x_pos,
                'y': 250 + y_offset,
                'rotation': 0,
                'properties': {
                    'waveplatetype': 'quarter' if i % 2 == 0 else 'half',
                    'angle': 45 + (i * 22.5),
                    'label': f'W{i+1}'
                }
            })
        
        if waveplate_count > 0:
            x_pos += 120

        for i in range(beamsplitter_count):
            y_offset = i * 100 - (beamsplitter_count - 1) * 50
            components.append({
                'id': f'beamsplitter_{i}',
                'type': 'beamsplitter',
                'x': x_pos,
                'y': 250 + y_offset,
                'rotation': 0,
                'properties': {
                    'reflectivity': 0.5,
                    'label': f'BS{i+1}'
                }
            })
        
        if beamsplitter_count > 0:
            x_pos += 120
        
        for i in range(mirror_count):
            y_offset = i * 100 - (mirror_count - 1) * 50
            components.append({
                'id': f'mirror_{i}',
                'type': 'mirror',
                'x': x_pos,
                'y': 250 + y_offset,
                'rotation': 45,
                'properties': {
                    'reflectivity': 0.98,
                    'label': f'M{i+1}'
                }
            })
        
        if mirror_count > 0:
            x_pos += 120
        
        detector_y_positions = np.linspace(150, 450, detector_count)
        for i in range(detector_count):
            components.append({
                'id': f'detector_{i}',
                'type': 'detector',
                'x': x_pos,
                'y': float(detector_y_positions[i]),
                'rotation': 0,
                'properties': {
                    'efficiency': min(0.98, goal.target_efficiency),
                    'label': f'D{i+1}'
                }
            })
        
        return components
    
    def _generate_fallback_layout(self, goal: DesignGoal) -> OpticalLayout:
        laser_count = goal.laser_range[1]
        detector_count = goal.detector_range[1]
        beamsplitter_count = min(1, goal.beamsplitter_range[1])
        
        components = self._generate_random_layout(goal, seed=0)
        
        expected_efficiency = self._estimate_efficiency({'components': components})
        expected_loss = self._estimate_loss({'components': components})
        
        return OpticalLayout(
            components=components,
            performance_score=50.0,
            detection_efficiency=expected_efficiency,
            photon_loss=expected_loss,
            provenance='fallback_heuristic',
            explanation=['Model not trained', 'Using fallback heuristics'],
            score_breakdown={}
        )
    
    def _estimate_transmission(self, layout: Dict) -> float:
        transmission = 1.0
        
        for comp in layout['components']:
            if comp['type'] == 'beamsplitter':
                transmission *= 0.5
            elif comp['type'] == 'polarizer':
                transmission *= 0.8
            elif comp['type'] == 'mirror':
                transmission *= 0.95
            elif comp['type'] == 'waveplate':
                transmission *= 0.98
        
        return max(0.0, min(1.0, transmission))
    
    def _estimate_efficiency(self, layout: Dict) -> float:
        detector_count = sum(1 for c in layout['components'] if c['type'] == 'detector')
        if detector_count == 0:
            return 0.0
        
        detector_efficiency = 0.85
        for comp in layout['components']:
            if comp['type'] == 'detector':
                detector_efficiency = max(detector_efficiency, comp['properties'].get('efficiency', 0.85))
        
        transmission = self._estimate_transmission(layout)
        return detector_efficiency * transmission
    
    def _estimate_loss(self, layout: Dict) -> float:
        transmission = self._estimate_transmission(layout)
        return 1.0 - transmission
    
    def save_model(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'trained_patterns': self.trained_patterns,
                'performance_history': self.performance_history
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
            self.trained_patterns = data['trained_patterns']
            self.performance_history = data['performance_history']
        print(f"Model loaded from {filepath}")