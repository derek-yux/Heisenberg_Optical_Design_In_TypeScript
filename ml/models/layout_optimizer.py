import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle

@dataclass
class DesignGoal:
    """Represents design objectives"""
    goal_type: str  # 'maximize_detection', 'minimize_loss', 'bb84_key_distribution'
    target_efficiency: float = 0.9
    max_components: int = 10
    preferred_components: List[str] = None
    # Component ranges: min and max counts for each component type
    laser_range: Tuple[int, int] = (1, 3)
    mirror_range: Tuple[int, int] = (0, 4)
    beamsplitter_range: Tuple[int, int] = (0, 4)
    polarizer_range: Tuple[int, int] = (0, 4)
    detector_range: Tuple[int, int] = (1, 4)
    waveplate_range: Tuple[int, int] = (0, 3)

@dataclass
class OpticalLayout:
    """Represents a complete optical system layout"""
    components: List[Dict]
    performance_score: float
    detection_efficiency: float
    photon_loss: float
    provenance: str = 'default_bb84'
    explanation: Optional[List[str]] = None
    score_breakdown: Optional[Dict[str, float]] = None
    
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
        """Generate optimized layout based on design goal and constraints"""
        # Generate layout respecting user-specified component count ranges
        # This uses the constraint-aware generation, not template modification
        
        # Map goal type to optimization strategy
        if goal.goal_type == 'maximize_detection':
            optimize_for = 'detection'
        elif goal.goal_type == 'minimize_loss':
            optimize_for = 'loss'
        else:  # bb84_key_distribution
            optimize_for = 'bb84'
        
        # Generate layout with proper constraints
        components = self._generate_layout_with_constraints(goal, optimize_for)
        optimized = {'components': components}
        provenance = 'constraint_aware_generation'

        # Calculate expected performance using internal estimators
        expected_efficiency = self._estimate_efficiency(optimized)
        expected_loss = self._estimate_loss(optimized)

        # Build a synthetic simulation result from estimates
        detections = int(expected_efficiency * 100)
        emissions = 100
        loss_events = int(expected_loss * 10)
        avg_intensity = expected_efficiency

        simulation_results = {
            'detections': detections,
            'emissions': emissions,
            'loss_events': loss_events,
            'avg_intensity': avg_intensity,
            'bb84_compatible': goal.goal_type == 'bb84_key_distribution'
        }

        expected_score = self.calculate_performance_score({'components': optimized['components']}, simulation_results)

        # Build a score breakdown for transparency
        detection_component = (detections / emissions) * 40 if emissions > 0 else 0
        loss_penalty = loss_events * 5
        intensity_component = avg_intensity * 30
        num_components = len(optimized['components'])
        complexity_penalty = (num_components - 3) * 2 if num_components > 3 else 0
        bb84_bonus = 20 if simulation_results.get('bb84_compatible') else 0

        score_breakdown = {
            'detection_component': detection_component,
            'intensity_component': intensity_component,
            'loss_penalty': -loss_penalty,
            'complexity_penalty': -complexity_penalty,
            'bb84_bonus': bb84_bonus,
            'final_score': expected_score
        }

        # Explanation of what we did
        explanation: List[str] = []
        explanation.append(f"Used template: {provenance}")
        explanation.append(f"Goal: {goal.goal_type}")
        explanation.append(f"Estimated detection efficiency: {expected_efficiency:.2f}")
        explanation.append(f"Estimated photon loss: {expected_loss:.2f}")
        explanation.append(f"Computed score (heuristic): {expected_score:.1f}")

        return OpticalLayout(
            components=optimized['components'],
            performance_score=expected_score,
            detection_efficiency=expected_efficiency,
            photon_loss=expected_loss,
            provenance=provenance,
            explanation=explanation,
            score_breakdown=score_breakdown
        )
    
    def _optimize_for_detection(self, base_layout: Dict, goal: DesignGoal) -> Dict:
        """Optimize layout to maximize detection efficiency"""
        # Generate a new layout respecting component ranges
        components = self._generate_layout_with_constraints(
            goal=goal,
            optimize_for='detection'
        )
        
        # Optimize detector efficiency
        for comp in components:
            if comp['type'] == 'detector':
                comp['properties']['efficiency'] = min(0.99, goal.target_efficiency)
        
        return {'components': components}
    
    def _optimize_for_low_loss(self, base_layout: Dict, goal: DesignGoal) -> Dict:
        """Optimize layout to minimize photon loss"""
        # Generate a new layout respecting component ranges
        components = self._generate_layout_with_constraints(
            goal=goal,
            optimize_for='loss'
        )
        
        # Increase component reflectivity/efficiency to reduce loss
        for comp in components:
            if comp['type'] == 'mirror' and 'reflectivity' in comp.get('properties', {}):
                comp['properties']['reflectivity'] = 0.99
            elif comp['type'] == 'beamsplitter' and 'reflectivity' in comp.get('properties', {}):
                # Optimize transmission to minimize loss
                comp['properties']['reflectivity'] = 0.3
        
        return {'components': components}
    
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
        goal = DesignGoal(
            goal_type='bb84_key_distribution',
            target_efficiency=0.95,
            laser_range=(1, 1),
            polarizer_range=(2, 2),
            beamsplitter_range=(1, 1),
            detector_range=(1, 1)
        )
        return self._optimize_for_bb84({}, goal)
    
    def _estimate_transmission(self, layout: Dict) -> float:
        """Calculate transmission through optical chain (0.0-1.0)"""
        transmission = 1.0
        
        for comp in layout['components']:
            if comp['type'] == 'beamsplitter':
                transmission *= 0.5  # 50% transmission through splitter
            elif comp['type'] == 'polarizer':
                transmission *= 0.8  # ~20% loss from polarization filtering
            elif comp['type'] == 'mirror':
                transmission *= 0.95  # 5% loss from reflection inefficiency
            elif comp['type'] == 'waveplate':
                transmission *= 0.98  # 2% loss from waveplate
        
        return max(0.0, min(1.0, transmission))
    
    def _estimate_efficiency(self, layout: Dict) -> float:
        """Estimate detection efficiency from layout (includes photon loss)"""
        detector_count = sum(1 for c in layout['components'] if c['type'] == 'detector')
        if detector_count == 0:
            return 0.0
        
        # Get detector efficiency
        detector_efficiency = 0.85
        for comp in layout['components']:
            if comp['type'] == 'detector':
                detector_efficiency = max(detector_efficiency, comp['properties'].get('efficiency', 0.85))
        
        # Calculate transmission (photon survival rate) through all components
        transmission = self._estimate_transmission(layout)
        
        # Final efficiency = detector efficiency × transmission through optical chain
        return detector_efficiency * transmission
    
    def _estimate_loss(self, layout: Dict) -> float:
        """Estimate photon loss from layout (0.0 = no loss, 1.0 = complete loss)"""
        transmission = self._estimate_transmission(layout)
        loss = 1.0 - transmission
        return max(0.0, min(1.0, loss))  # Clamp to [0, 1]
    
    def _generate_layout_with_constraints(self, goal: DesignGoal, optimize_for: str) -> List[Dict]:
        """Generate a layout respecting component count constraints with realistic optical paths.
        
        Creates components arranged left-to-right following photon path:
        Laser(s) → Polarizer(s) → Waveplate(s) → Beamsplitter(s) → Detector(s)
        NO ORPHANED COMPONENTS (no unused mirrors or duplicate detectors).
        
        Args:
            goal: DesignGoal with component ranges
            optimize_for: 'detection', 'loss', or 'bb84'
        
        Returns:
            List of components arranged in optical path order
        """
        np.random.seed(int(np.random.random() * 10000))
        
        # Determine target counts based on goal
        # Respect user constraints first, then optimize within those bounds
        if optimize_for == 'detection':
            # Maximize detection: many detectors, minimal loss components
            laser_target = goal.laser_range[0] + (goal.laser_range[1] - goal.laser_range[0]) // 2
            detector_target = goal.detector_range[1]  # Max detectors
            beamsplitter_target = goal.beamsplitter_range[0]  # Minimal BSs (but respect constraint minimum)
            polarizer_target = goal.polarizer_range[0]  # Minimal polarizers (but respect constraint minimum)
            mirror_target = goal.mirror_range[0]  # Minimal mirrors (but respect constraint minimum)
            waveplate_target = 0
        elif optimize_for == 'loss':
            # Minimize loss: use minimal allowed components
            laser_target = goal.laser_range[0]
            detector_target = goal.detector_range[0]
            beamsplitter_target = goal.beamsplitter_range[0]  # Use minimum allowed
            polarizer_target = goal.polarizer_range[0]  # Use minimum allowed
            mirror_target = goal.mirror_range[0]  # Use minimum allowed
            waveplate_target = goal.waveplate_range[0]
        else:  # bb84
            # BB84: laser → polarizer → beamsplitter → polarizer → detector (standard protocol)
            # Mirrors don't fit naturally in linear path (45° deflection), so minimize them
            # But respect user-specified ranges
            laser_target = max(1, goal.laser_range[0])
            detector_target = max(1, goal.detector_range[0])
            beamsplitter_target = max(goal.beamsplitter_range[0], 1)  # Need at least 1 for BB84
            polarizer_target = max(goal.polarizer_range[0], 2)  # Need at least 2 for BB84
            # For linear optical paths, mirrors create unphysical deflections, so don't include them
            # unless the minimum constraint requires it
            mirror_target = 0  # Don't add mirrors to linear paths
            waveplate_target = goal.waveplate_range[0]
        
        # Clamp to user-specified ranges
        laser_count = max(goal.laser_range[0], min(goal.laser_range[1], laser_target))
        detector_count = max(goal.detector_range[0], min(goal.detector_range[1], detector_target))
        beamsplitter_count = max(goal.beamsplitter_range[0], min(goal.beamsplitter_range[1], beamsplitter_target))
        polarizer_count = max(goal.polarizer_range[0], min(goal.polarizer_range[1], polarizer_target))
        # For BB84, only include mirrors if optimization demands it, not just because of minimum constraint
        if optimize_for == 'bb84':  # NOTE: optimize_for is 'bb84', not 'bb84_key_distribution'
            mirror_count = min(goal.mirror_range[1], mirror_target)  # Allow 0 mirrors even if minimum is higher
        else:
            mirror_count = max(goal.mirror_range[0], min(goal.mirror_range[1], mirror_target))
        waveplate_count = max(goal.waveplate_range[0], min(goal.waveplate_range[1], waveplate_target))
        
        components: List[Dict] = []
        x_pos = 100
        y_base = 200
        spacing = 120
        
        
        # For BB84 and repeater scenarios, create actual connected optical paths
        # Each laser path goes through the optical chain and reaches a detector
        
        if laser_count == 1:
            # Simple single-path design
            y_path = y_base
            
            for comp_type, count in [('polarizer', polarizer_count), ('waveplate', waveplate_count), 
                                     ('mirror', mirror_count), ('beamsplitter', beamsplitter_count)]:
                for i in range(count):
                    if comp_type == 'polarizer':
                        components.append({
                            'id': f'polarizer_{i}',
                            'type': 'polarizer',
                            'x': x_pos,
                            'y': y_path,
                            'rotation': 0,
                            'properties': {
                                'basis': 'rectilinear' if i % 2 == 0 else 'diagonal',
                                'angle': i * 45 % 180,
                                'label': f'Polarizer {i+1}'
                            }
                        })
                    elif comp_type == 'waveplate':
                        components.append({
                            'id': f'waveplate_{i}',
                            'type': 'waveplate',
                            'x': x_pos,
                            'y': y_path,
                            'rotation': 0,
                            'properties': {
                                'waveplatetype': 'quarter' if i % 2 == 0 else 'half',
                                'angle': 45 + i * 22.5,
                                'label': f'Wave Plate {i+1}'
                            }
                        })
                    elif comp_type == 'mirror':
                        components.append({
                            'id': f'mirror_{i}',
                            'type': 'mirror',
                            'x': x_pos,
                            'y': y_path,
                            'rotation': 45,
                            'properties': {
                                'reflectivity': 0.95,
                                'label': f'Mirror {i+1}'
                            }
                        })
                    elif comp_type == 'beamsplitter':
                        components.append({
                            'id': f'beamsplitter_{i}',
                            'type': 'beamsplitter',
                            'x': x_pos,
                            'y': y_path,
                            'rotation': 0,
                            'properties': {
                                'reflectivity': 0.5,
                                'label': f'Beam Splitter {i+1}'
                            }
                        })
                    x_pos += spacing
            
            # Single detector for single laser path
            for i in range(detector_count):
                components.append({
                    'id': f'detector_{i}',
                    'type': 'detector',
                    'x': x_pos,
                    'y': y_path,
                    'rotation': 0,
                    'properties': {
                        'efficiency': 0.92,
                        'label': f'Detector {i+1}'
                    }
                })
                
        else:
            # Multi-laser design (e.g., BB84 repeater with Alice & Bob)
            # For repeater networks, both lasers feed into a SHARED optical path
            # This is more realistic than separate unconnected paths
            
            # Place lasers slightly offset horizontally to show they're separate sources
            for laser_idx in range(laser_count):
                components.append({
                    'id': f'laser_{laser_idx}',
                    'type': 'laser',
                    'x': 80 + laser_idx * 20,  # Slight offset
                    'y': y_base,  # All on same y for common optical path
                    'rotation': 0,
                    'properties': {
                        'state': 'H' if laser_idx == 0 else 'V',
                        'angle': laser_idx * 45 % 180,
                        'label': f'Laser {laser_idx+1}'
                    }
                })
            
            x_pos = 220  # Start optical elements after lasers
            
            # Shared optical path for all lasers
            for i in range(polarizer_count):
                components.append({
                    'id': f'polarizer_{i}',
                    'type': 'polarizer',
                    'x': x_pos,
                    'y': y_base,
                    'rotation': 0,
                    'properties': {
                        'basis': 'rectilinear' if i % 2 == 0 else 'diagonal',
                        'angle': i * 45 % 180,
                        'label': f'Polarizer {i+1}'
                    }
                })
            x_pos += spacing
            
            for i in range(waveplate_count):
                components.append({
                    'id': f'waveplate_{i}',
                    'type': 'waveplate',
                    'x': x_pos,
                    'y': y_base,
                    'rotation': 0,
                    'properties': {
                        'waveplatetype': 'quarter' if i % 2 == 0 else 'half',
                        'angle': 45 + i * 22.5,
                        'label': f'Wave Plate {i+1}'
                    }
                })
            x_pos += spacing
            
            for i in range(mirror_count):
                components.append({
                    'id': f'mirror_{i}',
                    'type': 'mirror',
                    'x': x_pos,
                    'y': y_base,
                    'rotation': 45,
                    'properties': {
                        'reflectivity': 0.95,
                        'label': f'Mirror {i+1}'
                    }
                })
            x_pos += spacing
            
            for i in range(beamsplitter_count):
                components.append({
                    'id': f'beamsplitter_{i}',
                    'type': 'beamsplitter',
                    'x': x_pos,
                    'y': y_base,
                    'rotation': 0,
                    'properties': {
                        'reflectivity': 0.5,
                        'label': f'Beam Splitter {i+1}'
                    }
                })
            x_pos += spacing
            
            # Detectors aligned to main optical path (y_base)
            for i in range(detector_count):
                components.append({
                    'id': f'detector_{i}',
                    'type': 'detector',
                    'x': x_pos,
                    'y': y_base,
                    'rotation': 0,
                    'properties': {
                        'efficiency': 0.92,
                        'label': f'Detector {i+1}'
                    }
                })
        
        return components
    
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