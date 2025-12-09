# Heisenberg: The Quantum Optics Designer, by Derek (Yue) Yu

An interactive optical design system for quantum computing applications with a focus on quantum key distribution (BB84 protocol).

### 🤖 AI-Powered Layout Optimization
Click **ML Optimize** to auto-generate optical designs. Set component constraints (min/max counts for each type) and choose a goal:
- **BB84 Key Distribution** — balanced cryptography setup
- **Maximize Detection** — multi-detector arrays for high photon capture
- **Minimize Loss** — direct paths with minimal scattering

Get instant explanations, performance scores, and tailored topologies—no templates.

<img width="1896" height="865" alt="Screenshot 2025-12-03 at 10 35 10 PM" src="https://github.com/user-attachments/assets/ed9f0bbc-dca4-41bc-9d2f-b81f83dcd97e" />
<img width="1894" height="858" alt="Screenshot 2025-12-03 at 10 35 39 PM" src="https://github.com/user-attachments/assets/90c96cd9-de34-4e7f-8c23-d4261db44c24" />

For local development:
1. **Install frontend dependencies:**
```bash
npm install
```

2. **Start the ML optimization service (optional but recommended):**
```bash
cd ml
docker-compose -f docker-compose.ml.yml up -d
cd ..
```

3. **Start the development server:**
```bash
npm run dev
```
Then, open `http://localhost:5173`

With Docker:
```bash
# Start everything
docker-compose up -d --build

# Train model
docker exec quantum-ml-optimizer python scripts/train_model.py

# Stop everything
docker-compose down
```

Then, open `http://localhost:3000`

## Components Available

- **Laser Source**: Generates photons with specific polarization
- **Mirror**: Reflects light beams at angles
- **Beam Splitter**: Creates quantum superposition states
- **Polarizer**: Filters photons by polarization basis
- **Wave Plate**: Modifies photon polarization
- **Detector**: Measures photon arrival and state

## Components Library

| Component | Function | Editable Properties |
|-----------|----------|-------------------|
| **Laser Source** o | Generates photons with specific polarization | Polarization angle (0-180°) |
| **Mirror** 0  Reflects light beams at 90° angles | Reflectivity (50-100%) |
| **Beam Splitter** ◆ | Splits photons probabilistically | Reflectivity/Transmission ratio (0-100%) |
| **Polarizer** ⫴ | Filters photons by polarization basis | Polarization angle (0-180°) |
| **Wave Plate** ⊕ | Modifies photon quantum state | Quarter-wave (λ/4) or Half-wave (λ/2) |
| **Detector** X | Measures photon arrival and state | Detection efficiency (50-100%) |

## Usage

1. **Manual Design**: Drag components from the toolbar onto the canvas, reposition, and rotate
2. **AI-Guided**: Click "ML Optimize" to auto-generate designs based on goals and component constraints
3. **Simulate**: Click "Run Simulation" to trace photons through your optical system
4. **Demo**: Try "Load BB84 Demo" for a pre-built quantum cryptography example

## Use Case: Entanglement Distribution for Quantum Repeater Networks

### The Real Problem
You're building a **quantum repeater node** for long-distance quantum communication. Your system must:

1. **Receive entangled photon pairs** from two separate sources (Alice & Bob laser lines)
2. **Route beams optically** using mirrors (physical lab constraint—beams can't disappear!)
3. **Measure in multiple bases** (H/V polarization for each party)
4. **Manipulate entanglement** with beamsplitters for entanglement swapping
5. **Detect outcomes** across multiple channels

This isn't a simple laser→detector system—it's a **constrained optimization problem** where you can't remove components without breaking the protocol.

### Hardware Constraints (from your lab)
```json
{
  "laser_range": [2, 2],           // Exactly 2 entangled photon sources (FIXED)
  "mirror_range": [1, 3],          // 1-3 mirrors for beam routing (REQUIRED)
  "beamsplitter_range": [1, 2],    // 1-2 beamsplitters for manipulation
  "polarizer_range": [2, 4],       // 2-4 polarizers for basis measurement
  "detector_range": [2, 5],        // 2-5 detectors for outcome measurement
  "waveplate_range": [0, 2]        // 0-2 waveplates (optional)
}
```

### The Workflow
1. **Define constraints**: Specify exact min/max for each component based on your hardware
2. **Choose optimization goal**: "BB84 Key Distribution" (exploits +20 bonus), "Maximize Detection", or "Minimize Loss"
3. **Click ML Optimize**: Model generates topology **respecting all constraints** (~50ms)
4. **Review the breakdown**: See exactly where the 42.4 score comes from (detection 35.6, intensity 26.8, loss -30, complexity -10, BB84 +20)
5. **Run simulation**: Verify photon paths and measurement logic
6. **Iterate**: Adjust constraints (e.g., "try 2-3 mirrors instead of 1") and re-optimize to explore variations

### Training Data
The model learned from **7,171 realistic optical topologies** covering all combinations of:
- Lasers: 1-4 sources
- Mirrors: 0-4 for routing
- Beamsplitters: 0-3 for manipulation
- Polarizers: 0-4 for basis measurement
- Detectors: 1-6 for outcomes
- Waveplates: 0-2 for state control

This diversity enables the model to understand **constraint-respecting trade-offs** across all realistic quantum optics scenarios.

## Simulation Output

The simulation logs show:
- **Emission events** (blue): Photon generation from lasers
- **Interaction events** (purple): Component interactions with probability calculations
- **Detection events** (green): Successful photon measurements
- **Loss events** (red): Photon absorption or system exits

Each event includes:
- Intensity bars showing photon strength
- Probability calculations (P = %)
- Component labels and IDs

### Frontend
- **React 18** - UI framework
- **TypeScript 5** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Icon library

### Backend (ML Service)
- **Python 3.11** - Runtime
- **Flask** - REST API framework
- **NumPy** - Numerical computations
- **Docker** - Containerization

## License

MIT
