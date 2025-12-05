import React, { useState, useRef, useEffect } from 'react';
import { Zap, Move, Trash2, Play, RotateCw, Info, X, Copy, Settings, BarChart3 } from 'lucide-react';

type ComponentType = 'laser' | 'mirror' | 'beamsplitter' | 'polarizer' | 'detector' | 'waveplate';
type PolarizationBasis = 'rectilinear' | 'diagonal';
type PolarizationState = 'H' | 'V' | 'D' | 'A' | 'R' | 'L';

interface OpticalComponent {
  id: string;
  type: ComponentType;
  x: number;
  y: number;
  rotation: number;
  properties: {
    angle?: number;
    basis?: PolarizationBasis;
    state?: PolarizationState;
    label?: string;
    reflectivity?: number;
    efficiency?: number;
    waveplatetype?: 'quarter' | 'half';
  };
}

interface Beam {
  from: { x: number; y: number };
  to: { x: number; y: number };
  state: PolarizationState;
  intensity: number;
  probabilityAmplitude?: { real: number; imag: number };
}

interface SimulationLog {
  timestamp: number;
  type: 'emission' | 'interaction' | 'detection' | 'loss';
  message: string;
  componentId?: string;
  intensity?: number;
  state?: PolarizationState;
  probability?: number;
}

interface QuantumState {
  H: { real: number; imag: number };
  V: { real: number; imag: number };
}

const App = () => {
  const [components, setComponents] = useState<OpticalComponent[]>([]);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [dragging, setDragging] = useState<{ id: string; startX: number; startY: number; componentStartX: number; componentStartY: number } | null>(null);
  const [beams, setBeams] = useState<Beam[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [showInfo, setShowInfo] = useState(true);
  const [simulationLogs, setSimulationLogs] = useState<SimulationLog[]>([]);
  const [showLogs, setShowLogs] = useState(false);
  const [showProperties, setShowProperties] = useState(false);
  const [showProbabilities, setShowProbabilities] = useState(true);
  const [detectorResults, setDetectorResults] = useState<Map<string, { counts: Map<PolarizationState, number>; totalPhotons: number }>>(new Map());
  const canvasRef = useRef<HTMLDivElement>(null);
  const copiedComponents = useRef<OpticalComponent[]>([]);

  const componentLibrary: { type: ComponentType; icon: string; label: string; gradient: string }[] = [
    { type: 'laser', icon: '◉', label: 'Laser Source', gradient: 'from-red-500 to-orange-600' },
    { type: 'mirror', icon: '▱', label: 'Mirror', gradient: 'from-slate-400 to-slate-600' },
    { type: 'beamsplitter', icon: '◆', label: 'Beam Splitter', gradient: 'from-cyan-400 to-blue-600' },
    { type: 'polarizer', icon: '⫴', label: 'Polarizer', gradient: 'from-purple-500 to-pink-600' },
    { type: 'waveplate', icon: '⊕', label: 'Wave Plate', gradient: 'from-emerald-400 to-teal-600' },
    { type: 'detector', icon: '◎', label: 'Detector', gradient: 'from-amber-400 to-yellow-600' },
  ];

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedComponent) {
          setComponents(components.filter(c => c.id !== selectedComponent));
          setSelectedComponent(null);
          setShowProperties(false);
        }
      }
      
      if ((e.ctrlKey || e.metaKey) && e.key === 'c' && selectedComponent) {
        const component = components.find(c => c.id === selectedComponent);
        if (component) copiedComponents.current = [component];
      }
      
      if ((e.ctrlKey || e.metaKey) && e.key === 'v' && copiedComponents.current.length > 0) {
        const newComponents = copiedComponents.current.map(c => ({
          ...c,
          id: `${c.type}_${Date.now()}_${Math.random()}`,
          x: c.x + 50,
          y: c.y + 50,
        }));
        setComponents([...components, ...newComponents]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedComponent, components]);

  const loadBB84Demo = () => {
    const demoComponents: OpticalComponent[] = [
      { id: 'laser1', type: 'laser', x: 100, y: 200, rotation: 0, properties: { state: 'H', label: 'Alice', angle: 0 } },
      { id: 'polarizer1', type: 'polarizer', x: 200, y: 200, rotation: 0, properties: { basis: 'rectilinear', label: 'Alice Basis', angle: 0 } },
      { id: 'bs1', type: 'beamsplitter', x: 350, y: 200, rotation: 0, properties: { label: 'Channel', reflectivity: 0.5 } },
      { id: 'polarizer2', type: 'polarizer', x: 500, y: 200, rotation: 0, properties: { basis: 'rectilinear', label: 'Bob Basis', angle: 0 } },
      { id: 'detector1', type: 'detector', x: 600, y: 200, rotation: 0, properties: { label: 'Bob', efficiency: 0.95 } },
      { id: 'mirror1', type: 'mirror', x: 350, y: 300, rotation: 45, properties: { label: 'Eve', reflectivity: 1.0 } },
      { id: 'detector2', type: 'detector', x: 350, y: 380, rotation: 90, properties: { label: 'Eve Detector', efficiency: 0.90 } },
    ];
    setComponents(demoComponents);
    setShowInfo(true);
  };

  const addComponent = (type: ComponentType, x: number, y: number) => {
    const defaultProps: Record<ComponentType, any> = {
      laser: { state: 'H', angle: 0 },
      polarizer: { basis: 'rectilinear', angle: 0 },
      beamsplitter: { reflectivity: 0.5 },
      mirror: { reflectivity: 1.0 },
      detector: { efficiency: 0.95 },
      waveplate: { waveplatetype: 'quarter', angle: 45 },
    };

    const newComponent: OpticalComponent = {
      id: `${type}_${Date.now()}`,
      type,
      x,
      y,
      rotation: 0,
      properties: defaultProps[type],
    };
    setComponents([...components, newComponent]);
  };

  const updateComponentProperty = (id: string, property: string, value: any) => {
    setComponents(components.map(c =>
      c.id === id ? { ...c, properties: { ...c.properties, [property]: value } } : c
    ));
  };

  const calculateQuantumProbability = (state: QuantumState, angle: number): number => {
    const cosA = Math.cos(angle * Math.PI / 180);
    const sinA = Math.sin(angle * Math.PI / 180);
    
    const projectedReal = state.H.real * cosA + state.V.real * sinA;
    const projectedImag = state.H.imag * cosA + state.V.imag * sinA;
    
    return projectedReal * projectedReal + projectedImag * projectedImag;
  };

  const handleToolbarDragStart = (e: React.DragEvent, type: ComponentType) => {
    e.dataTransfer.setData('componentType', type);
  };

  const handleCanvasDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const componentType = e.dataTransfer.getData('componentType') as ComponentType;
    if (componentType && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      addComponent(componentType, x, y);
    }
  };

  const handleComponentClick = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setSelectedComponent(id);
    setShowProperties(true);
  };

  const handleComponentDragStart = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const component = components.find(c => c.id === id);
    if (component && canvasRef.current) {
      setDragging({
        id,
        startX: e.clientX,
        startY: e.clientY,
        componentStartX: component.x,
        componentStartY: component.y,
      });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragging) {
      const deltaX = e.clientX - dragging.startX;
      const deltaY = e.clientY - dragging.startY;
      
      setComponents(components.map(c =>
        c.id === dragging.id 
          ? { ...c, x: dragging.componentStartX + deltaX, y: dragging.componentStartY + deltaY } 
          : c
      ));
    }
  };

  const handleMouseUp = () => {
    setDragging(null);
  };

  const rotateComponent = () => {
    if (selectedComponent) {
      setComponents(components.map(c =>
        c.id === selectedComponent ? { ...c, rotation: (c.rotation + 45) % 360 } : c
      ));
    }
  };

  const deleteComponent = () => {
    if (selectedComponent) {
      setComponents(components.filter(c => c.id !== selectedComponent));
      setSelectedComponent(null);
      setShowProperties(false);
    }
  };

  const resetBoard = () => {
    setComponents([]);
    setSelectedComponent(null);
    setBeams([]);
    setSimulationLogs([]);
    setDetectorResults(new Map());
    setShowProperties(false);
    setShowLogs(false);
  };

  const duplicateComponent = () => {
    if (selectedComponent) {
      const component = components.find(c => c.id === selectedComponent);
      if (component) {
        const newComponent = {
          ...component,
          id: `${component.type}_${Date.now()}`,
          x: component.x + 50,
          y: component.y + 50,
        };
        setComponents([...components, newComponent]);
        setSelectedComponent(newComponent.id);
      }
    }
  };

  const simulate = () => {
    setIsSimulating(true);
    const newBeams: Beam[] = [];
    const logs: SimulationLog[] = [];
    let logId = 0;
    const detectorData = new Map<string, { counts: Map<PolarizationState, number>; totalPhotons: number }>();

    const addLog = (type: SimulationLog['type'], message: string, data?: Partial<SimulationLog>) => {
      logs.push({ timestamp: logId++, type, message, ...data });
    };

    const lasers = components.filter(c => c.type === 'laser');
    addLog('emission', `Quantum simulation initiated: ${lasers.length} photon source(s) detected`);

    lasers.forEach(laser => {
      let currentPos = { x: laser.x + 30, y: laser.y };
      let currentState: QuantumState = { 
        H: { real: 1, imag: 0 }, 
        V: { real: 0, imag: 0 } 
      };
      
      const laserAngle = laser.properties.angle || 0;
      if (laserAngle !== 0) {
        const cos = Math.cos(laserAngle * Math.PI / 180);
        const sin = Math.sin(laserAngle * Math.PI / 180);
        currentState = {
          H: { real: cos, imag: 0 },
          V: { real: sin, imag: 0 }
        };
      }

      let intensity = 1.0;
      let angle = laser.rotation;
      
      addLog('emission', `Photon emitted from "${laser.properties.label || laser.id}" at ${laserAngle}° polarization`, {
        componentId: laser.id,
        intensity: 1.0,
        probability: 1.0,
      });

      for (let step = 0; step < 20 && intensity > 0.01; step++) {
        const dx = Math.cos(angle * Math.PI / 180) * 50;
        const dy = Math.sin(angle * Math.PI / 180) * 50;
        const nextPos = { x: currentPos.x + dx, y: currentPos.y + dy };

        const hitComponent = components.find(c => {
          const dist = Math.sqrt((c.x - nextPos.x) ** 2 + (c.y - nextPos.y) ** 2);
          return dist < 30 && c.id !== laser.id;
        });

        if (hitComponent) {
          const stateLabel = Math.abs(currentState.H.real) > Math.abs(currentState.V.real) ? 'H' : 'V';
          newBeams.push({
            from: currentPos,
            to: { x: hitComponent.x, y: hitComponent.y },
            state: stateLabel as PolarizationState,
            intensity,
            probabilityAmplitude: currentState.H,
          });

          if (hitComponent.type === 'polarizer') {
            const polarizerAngle = hitComponent.properties.angle || 0;
            const passProbability = calculateQuantumProbability(currentState, polarizerAngle);
            
            if (Math.random() < passProbability) {
              const cos = Math.cos(polarizerAngle * Math.PI / 180);
              const sin = Math.sin(polarizerAngle * Math.PI / 180);
              currentState = {
                H: { real: cos, imag: 0 },
                V: { real: sin, imag: 0 }
              };
              intensity *= passProbability;
              
              addLog('interaction', `Polarizer "${hitComponent.properties.label || hitComponent.id}" (${polarizerAngle}°): Photon passed with probability ${(passProbability * 100).toFixed(1)}%`, {
                componentId: hitComponent.id,
                intensity,
                probability: passProbability,
              });
            } else {
              addLog('loss', `Polarizer "${hitComponent.properties.label || hitComponent.id}" absorbed photon (P=${((1-passProbability) * 100).toFixed(1)}%)`, {
                componentId: hitComponent.id,
                probability: 1 - passProbability,
              });
              break;
            }
            currentPos = { x: hitComponent.x + 30, y: hitComponent.y };
          } else if (hitComponent.type === 'beamsplitter') {
            const reflectivity = hitComponent.properties.reflectivity || 0.5;
            const transmitted = Math.random() < (1 - reflectivity);
            
            if (transmitted) {
              intensity *= (1 - reflectivity);
              addLog('interaction', `Beam splitter: Photon transmitted (T=${((1-reflectivity) * 100).toFixed(0)}%)`, {
                componentId: hitComponent.id,
                intensity,
                probability: 1 - reflectivity,
              });
              currentPos = { x: hitComponent.x + 30, y: hitComponent.y };
            } else {
              intensity *= reflectivity;
              angle = (angle + 90) % 360;
              addLog('interaction', `Beam splitter: Photon reflected (R=${(reflectivity * 100).toFixed(0)}%)`, {
                componentId: hitComponent.id,
                intensity,
                probability: reflectivity,
              });
              currentPos = { x: hitComponent.x, y: hitComponent.y + 30 };
            }
          } else if (hitComponent.type === 'mirror') {
            angle = (angle + 90) % 360;
            const reflectivity = hitComponent.properties.reflectivity || 1.0;
            intensity *= reflectivity;
            addLog('interaction', `Mirror reflection (${(reflectivity * 100).toFixed(0)}% efficient)`, {
              componentId: hitComponent.id,
              intensity,
            });
            currentPos = { x: hitComponent.x + 30, y: hitComponent.y };
          } else if (hitComponent.type === 'waveplate') {
            const wpType = hitComponent.properties.waveplatetype || 'quarter';
            if (wpType === 'quarter') {
              const temp = currentState.H;
              currentState.H = currentState.V;
              currentState.V = { real: -temp.real, imag: -temp.imag };
              addLog('interaction', `Quarter-wave plate: H↔V transformation applied`, {
                componentId: hitComponent.id,
                intensity,
              });
            } else {
              currentState.V = { real: -currentState.V.real, imag: -currentState.V.imag };
              addLog('interaction', `Half-wave plate: Phase shift applied`, {
                componentId: hitComponent.id,
                intensity,
              });
            }
            currentPos = { x: hitComponent.x + 30, y: hitComponent.y };
          } else if (hitComponent.type === 'detector') {
            const efficiency = hitComponent.properties.efficiency || 0.95;
            if (Math.random() < efficiency) {
              const measuredState = Math.abs(currentState.H.real) > Math.abs(currentState.V.real) ? 'H' : 'V';
              
              if (!detectorData.has(hitComponent.id)) {
                detectorData.set(hitComponent.id, { counts: new Map(), totalPhotons: 0 });
              }
              const data = detectorData.get(hitComponent.id)!;
              data.counts.set(measuredState, (data.counts.get(measuredState) || 0) + 1);
              data.totalPhotons++;
              
              addLog('detection', `✓ Detector "${hitComponent.properties.label || hitComponent.id}" registered ${measuredState} polarization (η=${(efficiency * 100).toFixed(0)}%)`, {
                componentId: hitComponent.id,
                state: measuredState,
                intensity,
                probability: efficiency,
              });
            } else {
              addLog('loss', `Detector missed photon (inefficiency: ${((1-efficiency) * 100).toFixed(0)}%)`, {
                componentId: hitComponent.id,
              });
            }
            break;
          } else {
            currentPos = { x: hitComponent.x + 30, y: hitComponent.y };
          }
        } else {
          const stateLabel = Math.abs(currentState.H.real) > Math.abs(currentState.V.real) ? 'H' : 'V';
          newBeams.push({
            from: currentPos,
            to: nextPos,
            state: stateLabel as PolarizationState,
            intensity,
            probabilityAmplitude: currentState.H,
          });
          currentPos = nextPos;
        }

        if (nextPos.x < 0 || nextPos.x > 1000 || nextPos.y < 0 || nextPos.y > 800) {
          addLog('loss', `Photon exited optical system`, { intensity });
          break;
        }
      }
    });

    addLog('emission', `Simulation complete: ${logs.filter(l => l.type === 'detection').length} detection events recorded`);

    setBeams(newBeams);
    setSimulationLogs(logs);
    setDetectorResults(detectorData);
    setShowLogs(true);
    setTimeout(() => setIsSimulating(false), 500);
  };

  const renderComponent = (component: OpticalComponent) => {
    const libItem = componentLibrary.find(l => l.type === component.type);
    const isSelected = selectedComponent === component.id;

    return (
      <div
        key={component.id}
        className={`absolute cursor-move transition-all duration-200 ${isSelected ? 'scale-110 z-10' : 'hover:scale-105'}`}
        style={{
          left: component.x,
          top: component.y,
          transform: `translate(-50%, -50%) rotate(${component.rotation}deg)`,
        }}
        onClick={(e) => handleComponentClick(e, component.id)}
        onMouseDown={(e) => handleComponentDragStart(e, component.id)}
      >
        <div className={`relative w-14 h-14 rounded-xl bg-gradient-to-br ${libItem?.gradient} flex items-center justify-center text-2xl shadow-2xl backdrop-blur-sm border border-white/20 ${isSelected ? 'ring-2 ring-blue-400 ring-offset-2 ring-offset-transparent' : ''}`}>
          <div className="absolute inset-0 rounded-xl bg-white/10 backdrop-blur-md"></div>
          <span className="relative z-10 text-white drop-shadow-lg font-light">{libItem?.icon}</span>
        </div>
        {component.properties.label && (
          <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 text-xs whitespace-nowrap bg-gradient-to-r from-slate-900/90 to-slate-800/90 backdrop-blur-md text-slate-200 px-3 py-1.5 rounded-full border border-slate-700/50 shadow-lg">
            {component.properties.label}
          </div>
        )}
      </div>
    );
  };

  const selectedComp = components.find(c => c.id === selectedComponent);

  return (
    <div className="w-full h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900/80 to-slate-800/80 backdrop-blur-xl text-white p-5 flex items-center justify-between border-b border-slate-700/50 shadow-2xl">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
            <Zap size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-light tracking-wide">Heisenberg: The Quantum Optical Designer</h1>
            <p className="text-xs text-slate-400 font-light">Optical Systems Design CAD</p>
          </div>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowProbabilities(!showProbabilities)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all border backdrop-blur-sm text-sm font-light ${
              showProbabilities 
                ? 'bg-blue-600/50 border-blue-500/50' 
                : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-700/50'
            }`}
          >
            <BarChart3 size={14} />
            Probabilities
          </button>
          <button
            onClick={() => setShowInfo(!showInfo)}
            className="px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg flex items-center gap-2 transition-all border border-slate-700/50 backdrop-blur-sm text-sm font-light"
          >
            <Info size={14} />
            Info
          </button>
          <button
            onClick={resetBoard}
            className="px-4 py-2 bg-red-800/50 hover:bg-red-900/50 rounded-lg flex items-center gap-2 transition-all border border-slate-700/50 hover:border-red-700/50 backdrop-blur-sm text-sm font-light text-slate-300 hover:text-red-300"
          >
            <Trash2 size={14} />
            Reset
          </button>
          <button
            onClick={loadBB84Demo}
            className="px-4 py-2 bg-gradient-to-r from-purple-600/80 to-pink-600/80 hover:from-purple-500/80 hover:to-pink-500/80 rounded-lg transition-all shadow-lg text-sm font-light"
          >
            Load BB84 Demo
          </button>
          <button
            onClick={simulate}
            disabled={isSimulating}
            className="px-5 py-2 bg-gradient-to-r from-emerald-600/80 to-teal-600/80 hover:from-emerald-500/80 hover:to-teal-500/80 rounded-lg flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-lg text-sm font-light"
          >
            <Play size={14} />
            {isSimulating ? 'Simulating...' : 'Run Simulation'}
          </button>
        </div>
      </div>

      {/* Info Panel */}
      {showInfo && (
        <div className="relative bg-gradient-to-r from-blue-950/40 to-purple-950/40 backdrop-blur-md text-slate-200 p-5 border-b border-slate-800/50">
          <button 
            onClick={() => setShowInfo(false)}
            className="absolute top-3 right-3 p-1 hover:bg-white/10 rounded-lg transition-colors"
          >
            <X size={16} />
          </button>
          <h3 className="font-light text-sm mb-2 text-blue-300">Your Solution for Quantum Optics Systems Simulation</h3>
          <p className="text-xs text-slate-300 mb-3 font-light leading-relaxed max-w-4xl">
            Full quantum mechanical simulation with probability amplitudes, detector statistics, and editable component parameters. 
            Click any component to edit its properties in real-time.
          </p>
          <div className="text-xs space-y-1 text-slate-400 font-light">
            <p><strong>Features:</strong> Quantum probability calculations • Detector efficiency modeling • Adjustable polarization angles • Statistical analysis • Wave plate transformations</p>
          </div>
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Toolbar */}
        <div className="w-56 bg-gradient-to-b from-slate-900/50 to-slate-900/30 backdrop-blur-xl p-5 border-r border-slate-800/50 overflow-y-auto">
          <h2 className="text-slate-300 font-light text-sm mb-5 tracking-wide">COMPONENTS</h2>
          <div className="space-y-3">
            {componentLibrary.map((item) => (
              <div
                key={item.type}
                draggable
                onDragStart={(e) => handleToolbarDragStart(e, item.type)}
                className={`relative overflow-hidden bg-gradient-to-br ${item.gradient} p-4 rounded-xl cursor-move hover:scale-105 transition-all duration-200 shadow-lg border border-white/10 backdrop-blur-sm group`}
              >
                <div className="absolute inset-0 bg-white/5 backdrop-blur-sm"></div>
                <div className="relative z-10">
                  <div className="text-3xl mb-2 text-white drop-shadow-lg font-light">{item.icon}</div>
                  <div className="text-xs text-white/90 font-light tracking-wide">{item.label}</div>
                </div>
              </div>
            ))}
          </div>

          {selectedComponent && (
            <div className="mt-8 pt-5 border-t border-slate-800/50">
              <h3 className="text-slate-300 font-light text-sm mb-3 tracking-wide">ACTIONS</h3>
              <div className="space-y-2">
                <button
                  onClick={() => setShowProperties(!showProperties)}
                  className="w-full px-4 py-2.5 bg-blue-950/30 hover:bg-blue-900/40 text-blue-300 rounded-lg flex items-center justify-center gap-2 transition-all border border-blue-800/30 text-sm font-light"
                >
                  <Settings size={14} />
                  Properties
                </button>
                <button
                  onClick={rotateComponent}
                  className="w-full px-4 py-2.5 bg-slate-800/50 hover:bg-slate-700/50 text-slate-200 rounded-lg flex items-center justify-center gap-2 transition-all border border-slate-700/50 text-sm font-light"
                >
                  <RotateCw size={14} />
                  Rotate 45°
                </button>
                <button
                  onClick={duplicateComponent}
                  className="w-full px-4 py-2.5 bg-slate-800/50 hover:bg-slate-700/50 text-slate-200 rounded-lg flex items-center justify-center gap-2 transition-all border border-slate-700/50 text-sm font-light"
                >
                  <Copy size={14} />
                  Duplicate
                </button>
                <button
                  onClick={deleteComponent}
                  className="w-full px-4 py-2.5 bg-red-950/30 hover:bg-red-900/40 text-red-300 rounded-lg flex items-center justify-center gap-2 transition-all border border-red-800/30 text-sm font-light"
                >
                  <Trash2 size={14} />
                  Delete
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Canvas */}
        <div
          ref={canvasRef}
          className="flex-1 relative bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 overflow-hidden"
          onDrop={handleCanvasDrop}
          onDragOver={(e) => e.preventDefault()}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={() => { setSelectedComponent(null); setShowProperties(false); }}
        >
          {/* Grid */}
          <div 
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage: 'radial-gradient(circle, #60a5fa 1px, transparent 1px)',
              backgroundSize: '40px 40px'
            }}
          />

          {/* Beams with enhanced visualization */}
          <svg className="absolute inset-0 pointer-events-none" style={{ width: '100%', height: '100%' }}>
            <defs>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <linearGradient id="beamGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#ef4444" stopOpacity="0.3"/>
                <stop offset="50%" stopColor="#ef4444" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#ef4444" stopOpacity="0.3"/>
              </linearGradient>
            </defs>
            {beams.map((beam, i) => (
              <g key={i}>
                <line
                  x1={beam.from.x}
                  y1={beam.from.y}
                  x2={beam.to.x}
                  y2={beam.to.y}
                  stroke={beam.state === 'H' || beam.state === 'V' ? '#ef4444' : '#a78bfa'}
                  strokeWidth="3"
                  opacity={beam.intensity * 0.7}
                  filter="url(#glow)"
                  className="animate-pulse"
                />
                {showProbabilities && (
                  <text
                    x={(beam.from.x + beam.to.x) / 2}
                    y={(beam.from.y + beam.to.y) / 2 - 10}
                    fill="#60a5fa"
                    fontSize="10"
                    fontFamily="monospace"
                    textAnchor="middle"
                  >
                    I={beam.intensity.toFixed(2)}
                  </text>
                )}
                <circle
                  cx={beam.to.x}
                  cy={beam.to.y}
                  r="4"
                  fill={beam.state === 'H' || beam.state === 'V' ? '#ef4444' : '#a78bfa'}
                  opacity={beam.intensity}
                  filter="url(#glow)"
                />
              </g>
            ))}
          </svg>

          {/* Components */}
          {components.map(renderComponent)}

          {/* Detector Statistics Overlay */}
          {showProbabilities && detectorResults.size > 0 && components.filter(c => c.type === 'detector').map(detector => {
            const stats = detectorResults.get(detector.id);
            if (!stats || stats.totalPhotons === 0) return null;
            
            return (
              <div
                key={detector.id}
                className="absolute pointer-events-none"
                style={{ left: detector.x + 40, top: detector.y - 60 }}
              >
                <div className="bg-slate-900/95 backdrop-blur-md border border-emerald-500/50 rounded-lg p-3 shadow-2xl min-w-[140px]">
                  <div className="text-xs font-light text-emerald-400 mb-2">Detection Stats</div>
                  <div className="space-y-1">
                    {Array.from(stats.counts.entries()).map(([state, count]) => {
                      const probability = count / stats.totalPhotons;
                      return (
                        <div key={state} className="flex items-center gap-2">
                          <div className="text-xs text-slate-300 w-6">{state}:</div>
                          <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                              style={{ width: `${probability * 100}%` }}
                            />
                          </div>
                          <div className="text-xs text-slate-400 w-12 text-right">
                            {(probability * 100).toFixed(0)}%
                          </div>
                        </div>
                      );
                    })}
                    <div className="text-xs text-slate-500 mt-2 pt-2 border-t border-slate-800">
                      n={stats.totalPhotons}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Empty state */}
          {components.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center text-slate-600">
                <Move size={40} className="mx-auto mb-4 opacity-30" strokeWidth={1} />
                <p className="text-lg font-light">Drag components to canvas</p>
                <p className="text-sm mt-2 font-light opacity-60">or load the BB84 demo</p>
              </div>
            </div>
          )}
        </div>

        {/* Properties Panel */}
        {showProperties && selectedComp && (
          <div className="w-80 bg-gradient-to-b from-slate-900/50 to-slate-900/30 backdrop-blur-xl border-l border-slate-800/50 flex flex-col overflow-y-auto">
            <div className="p-5 border-b border-slate-800/50 flex items-center justify-between sticky top-0 bg-slate-900/80 backdrop-blur-xl z-10">
              <h2 className="text-slate-300 font-light text-sm tracking-wide">PROPERTIES</h2>
              <button 
                onClick={() => setShowProperties(false)}
                className="p-1 hover:bg-white/10 rounded-lg transition-colors text-slate-400"
              >
                <X size={16} />
              </button>
            </div>
            <div className="p-5 space-y-4">
              <div>
                <label className="text-xs text-slate-400 font-light mb-2 block">Component Type</label>
                <div className="text-sm text-slate-200 font-mono bg-slate-800/50 px-3 py-2 rounded-lg border border-slate-700/50">
                  {selectedComp.type}
                </div>
              </div>

              <div>
                <label className="text-xs text-slate-400 font-light mb-2 block">Label</label>
                <input
                  type="text"
                  value={selectedComp.properties.label || ''}
                  onChange={(e) => updateComponentProperty(selectedComp.id, 'label', e.target.value)}
                  className="w-full bg-slate-800/50 text-slate-200 px-3 py-2 rounded-lg border border-slate-700/50 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  placeholder="Enter label..."
                />
              </div>

              {selectedComp.type === 'laser' && (
                <div>
                  <label className="text-xs text-slate-400 font-light mb-2 block">Polarization Angle (°)</label>
                  <input
                    type="range"
                    min="0"
                    max="180"
                    value={selectedComp.properties.angle || 0}
                    onChange={(e) => updateComponentProperty(selectedComp.id, 'angle', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-slate-400 mt-1">
                    <span>0° (H)</span>
                    <span className="text-blue-400 font-mono">{selectedComp.properties.angle || 0}°</span>
                    <span>90° (V)</span>
                  </div>
                  <div className="mt-3 bg-slate-800/50 rounded-lg p-3 border border-slate-700/50">
                    <div className="text-xs text-slate-400 mb-2">Visual representation:</div>
                    <svg width="100%" height="60" className="bg-slate-900/50 rounded">
                      <line x1="10" y1="30" x2="150" y2="30" stroke="#334155" strokeWidth="1" />
                      <line 
                        x1="80" 
                        y1="30" 
                        x2={80 + 30 * Math.cos((selectedComp.properties.angle || 0) * Math.PI / 180)} 
                        y2={30 - 30 * Math.sin((selectedComp.properties.angle || 0) * Math.PI / 180)}
                        stroke="#ef4444" 
                        strokeWidth="3"
                        markerEnd="url(#arrowhead)"
                      />
                      <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="5" refY="3" orient="auto">
                          <polygon points="0 0, 5 3, 0 6" fill="#ef4444" />
                        </marker>
                      </defs>
                    </svg>
                  </div>
                </div>
              )}

              {selectedComp.type === 'polarizer' && (
                <>
                  <div>
                    <label className="text-xs text-slate-400 font-light mb-2 block">Polarization Angle (°)</label>
                    <input
                      type="range"
                      min="0"
                      max="180"
                      value={selectedComp.properties.angle || 0}
                      onChange={(e) => updateComponentProperty(selectedComp.id, 'angle', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-slate-400 mt-1">
                      <span>0°</span>
                      <span className="text-purple-400 font-mono">{selectedComp.properties.angle || 0}°</span>
                      <span>180°</span>
                    </div>
                  </div>
                  <div className="bg-blue-950/30 border border-blue-800/30 rounded-lg p-3">
                    <div className="text-xs text-blue-300 font-light">
                      <strong>Quantum Mechanics:</strong> Malus's Law applies. Transmission probability = cos²(θ), where θ is the angle difference between photon and polarizer.
                    </div>
                  </div>
                </>
              )}

              {selectedComp.type === 'beamsplitter' && (
                <div>
                  <label className="text-xs text-slate-400 font-light mb-2 block">Reflectivity</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={selectedComp.properties.reflectivity || 0.5}
                    onChange={(e) => updateComponentProperty(selectedComp.id, 'reflectivity', parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-slate-400 mt-1">
                    <span>0% (All transmitted)</span>
                    <span className="text-cyan-400 font-mono">{((selectedComp.properties.reflectivity || 0.5) * 100).toFixed(0)}%</span>
                    <span>100% (All reflected)</span>
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <div className="bg-slate-800/50 rounded p-2 border border-slate-700/50">
                      <div className="text-xs text-slate-500">Transmission</div>
                      <div className="text-lg text-emerald-400 font-mono">{((1 - (selectedComp.properties.reflectivity || 0.5)) * 100).toFixed(0)}%</div>
                    </div>
                    <div className="bg-slate-800/50 rounded p-2 border border-slate-700/50">
                      <div className="text-xs text-slate-500">Reflection</div>
                      <div className="text-lg text-blue-400 font-mono">{((selectedComp.properties.reflectivity || 0.5) * 100).toFixed(0)}%</div>
                    </div>
                  </div>
                </div>
              )}

              {selectedComp.type === 'mirror' && (
                <div>
                  <label className="text-xs text-slate-400 font-light mb-2 block">Reflectivity</label>
                  <input
                    type="range"
                    min="0.5"
                    max="1"
                    step="0.01"
                    value={selectedComp.properties.reflectivity || 1.0}
                    onChange={(e) => updateComponentProperty(selectedComp.id, 'reflectivity', parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm text-slate-300 mt-2 font-mono">
                    {((selectedComp.properties.reflectivity || 1.0) * 100).toFixed(1)}%
                  </div>
                </div>
              )}

              {selectedComp.type === 'detector' && (
                <div>
                  <label className="text-xs text-slate-400 font-light mb-2 block">Detection Efficiency</label>
                  <input
                    type="range"
                    min="0.5"
                    max="1"
                    step="0.01"
                    value={selectedComp.properties.efficiency || 0.95}
                    onChange={(e) => updateComponentProperty(selectedComp.id, 'efficiency', parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-center text-sm text-slate-300 mt-2 font-mono">
                    η = {((selectedComp.properties.efficiency || 0.95) * 100).toFixed(1)}%
                  </div>
                  <div className="mt-3 bg-amber-950/30 border border-amber-800/30 rounded-lg p-3">
                    <div className="text-xs text-amber-300 font-light">
                      Real detectors have quantum efficiency &lt;100%. Some photons are not detected due to material absorption or reflection.
                    </div>
                  </div>
                </div>
              )}

              {selectedComp.type === 'waveplate' && (
                <div>
                  <label className="text-xs text-slate-400 font-light mb-2 block">Wave Plate Type</label>
                  <select
                    value={selectedComp.properties.waveplatetype || 'quarter'}
                    onChange={(e) => updateComponentProperty(selectedComp.id, 'waveplatetype', e.target.value)}
                    className="w-full bg-slate-800/50 text-slate-200 px-3 py-2 rounded-lg border border-slate-700/50 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  >
                    <option value="quarter">Quarter-wave (λ/4)</option>
                    <option value="half">Half-wave (λ/2)</option>
                  </select>
                  <div className="mt-3 bg-emerald-950/30 border border-emerald-800/30 rounded-lg p-3">
                    <div className="text-xs text-emerald-300 font-light">
                      {selectedComp.properties.waveplatetype === 'quarter' 
                        ? 'Converts linear → circular polarization (90° phase shift)'
                        : 'Rotates linear polarization (180° phase shift)'}
                    </div>
                  </div>
                </div>
              )}

              <div className="pt-4 border-t border-slate-800/50">
                <div className="text-xs text-slate-500 font-light space-y-1">
                  <div>Position: ({selectedComp.x.toFixed(0)}, {selectedComp.y.toFixed(0)})</div>
                  <div>Rotation: {selectedComp.rotation}°</div>
                  <div>ID: <span className="font-mono text-[10px]">{selectedComp.id}</span></div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Simulation Output Panel */}
        {showLogs && (
          <div className="w-96 bg-gradient-to-b from-slate-900/50 to-slate-900/30 backdrop-blur-xl border-l border-slate-800/50 flex flex-col">
            <div className="p-5 border-b border-slate-800/50 flex items-center justify-between">
              <h2 className="text-slate-300 font-light text-sm tracking-wide">SIMULATION LOG</h2>
              <button 
                onClick={() => setShowLogs(false)}
                className="p-1 hover:bg-white/10 rounded-lg transition-colors text-slate-400"
              >
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-5 space-y-3">
              {simulationLogs.map((log) => (
                <div 
                  key={log.timestamp}
                  className={`p-3 rounded-lg border text-xs font-light ${
                    log.type === 'emission' ? 'bg-blue-950/30 border-blue-800/30 text-blue-300' :
                    log.type === 'interaction' ? 'bg-purple-950/30 border-purple-800/30 text-purple-300' :
                    log.type === 'detection' ? 'bg-green-950/30 border-green-800/30 text-green-300' :
                    'bg-red-950/30 border-red-800/30 text-red-300'
                  }`}
                >
                  <div className="font-semibold mb-1 text-xs uppercase tracking-wider opacity-60">
                    {log.type}
                  </div>
                  <div className="leading-relaxed">{log.message}</div>
                  {log.intensity !== undefined && (
                    <div className="mt-2 flex items-center gap-2">
                      <div className="flex-1 bg-slate-900/50 rounded-full h-1.5 overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                          style={{ width: `${log.intensity * 100}%` }}
                        />
                      </div>
                      <span className="text-xs opacity-60">{(log.intensity * 100).toFixed(0)}%</span>
                    </div>
                  )}
                  {log.probability !== undefined && log.probability < 1 && (
                    <div className="mt-2 flex items-center gap-2">
                      <div className="text-[10px] text-slate-400">P:</div>
                      <div className="flex-1 bg-slate-900/50 rounded-full h-1.5 overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                          style={{ width: `${log.probability * 100}%` }}
                        />
                      </div>
                      <span className="text-xs opacity-60">{(log.probability * 100).toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;