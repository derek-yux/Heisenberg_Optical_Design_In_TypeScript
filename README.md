# Quantum Optics Designer

An interactive optical design system for quantum computing applications with a focus on quantum key distribution (BB84 protocol).

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open `http://localhost:5173`

## Components Available

- **Laser Source**: Generates photons with specific polarization
- **Mirror**: Reflects light beams at angles
- **Beam Splitter**: Creates quantum superposition states
- **Polarizer**: Filters photons by polarization basis
- **Wave Plate**: Modifies photon polarization
- **Detector**: Measures photon arrival and state

## Usage

1. Drag components from the toolbar onto the canvas
2. Click and drag components to reposition them
3. Select a component and use "Rotate" to change orientation
4. Click "Simulate" to see photon propagation
5. Try "Load BB84 Demo" for a pre-built quantum cryptography setup

## Tech Stack

- React 18
- TypeScript 5
- Vite
- Tailwind CSS
- Lucide React Icons

## License

MIT