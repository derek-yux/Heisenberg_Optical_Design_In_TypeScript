import React, { useState, useRef, useEffect } from 'react';
import { Zap, Move, Trash2, Play, RotateCw, Info, X, Copy } from 'lucide-react';

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
  };
}

interface Beam {
  from: { x: number; y: number };
  to: { x: number; y: number };
  state: PolarizationState;
  intensity: number;
}

interface SimulationLog {
  timestamp: number;
  type: 'emission' | 'interaction' | 'detection' | 'loss';
  message: string;
  componentId?: string;
  intensity?: number;
  state?: PolarizationState;
}

const QuantumOpticsDesigner = () => {
  const [components, setComponents] = useState<OpticalComponent[]>([]);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [selectedComponents, setSelectedComponents] = useState<string[]>([]);
  const [dragging, setDragging] = useState<{ id: string; startX: number; startY: number; componentStartX: number; componentStartY: number } | null>(null);
  const [beams, setBeams] = useState<Beam[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [showInfo, setShowInfo] = useState(true);
  const [simulationLogs, setSimulationLogs] = useState<SimulationLog[]>([]);
  const [showLogs, setShowLogs] = useState(false);
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

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Delete selected component
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedComponent) {
          setComponents(components.filter(c => c.id !== selectedComponent));
          setSelectedComponent(null);
        } else if (selectedComponents.length > 0) {
          setComponents(components.filter(c => !selectedComponents.includes(c.id)));
          setSelectedComponents([]);
        }
      }
      
      // Copy: Ctrl+C or Cmd+C
      if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        if (selectedComponents.length > 0) {
          copiedComponents.current = components.filter(c => selectedComponents.includes(c.id));
          console.log('Copied', copiedComponents.current.length, 'components');
        } else if (selectedComponent) {
          const component = components.find(c => c.id === selectedComponent);
          if (component) {
            copiedComponents.current = [component];
            console.log('Copied 1 component');
          }
        }
      }
      
      // Paste: Ctrl+V or Cmd+V
      if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
        if (copiedComponents.current.length > 0) {
          const newComponents = copiedComponents.current.map(c => ({
            ...c,
            id: `${c.type}_${Date.now()}_${Math.random()}`,
            x: c.x + 50,
            y: c.y + 50,
          }));
          setComponents([...components, ...newComponents]);
          setSelectedComponents(newComponents.map(c => c.id));
          console.log('Pasted', newComponents.length, 'components');
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedComponent, selectedComponents, components]);

  const loadBB84Demo = () => {
    const demoComponents: OpticalComponent[] = [
      { id: 'laser1', type: 'laser', x: 100, y: 200, rotation: 0, properties: { state: 'H', label: 'Alice' } },
      { id: 'polarizer1', type: 'polarizer', x: 200, y: 200, rotation: 0, properties: { basis: 'rectilinear', label: 'Basis' } },
      { id: 'bs1', type: 'beamsplitter', x: 350, y: 200, rotation: 0, properties: { label: 'Channel' } },
      { id: 'polarizer2', type: 'polarizer', x: 500, y: 200, rotation: 0, properties: { basis: 'rectilinear', label: 'Basis' } },
      { id: 'detector1', type: 'detector', x: 600, y: 200, rotation: 0, properties: { label: 'Bob' } },
      { id: 'mirror1', type: 'mirror', x: 350, y: 300, rotation: 45, properties: { label: 'Eve' } },
      { id: 'detector2', type: 'detector', x: 350, y: 380, rotation: 90, properties: { label: 'Intercept' } },
    ];
    setComponents(demoComponents);
    setShowInfo(true);
  };

  const addComponent = (type: ComponentType, x: number, y: number) => {
    const newComponent: OpticalComponent = {
      id: `${type}_${Date.now()}`,
      type,
      x,
      y,
      rotation: 0,
      properties: {
        state: type === 'laser' ? 'H' : undefined,
        basis: type === 'polarizer' ? 'rectilinear' : undefined,
      },
    };
    setComponents([...components, newComponent]);
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

  const handleComponentMouseDown = (e: React.MouseEvent, id: string) => {
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
      setSelectedComponent(id);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragging && canvasRef.current) {
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
    }
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

    const addLog = (type: SimulationLog['type'], message: string, data?: Partial<SimulationLog>) => {
      logs.push({
        timestamp: logId++,
        type,
        message,
        ...data,
      });
    };

    // Find all lasers and trace their beams
    const lasers = components.filter(c => c.type === 'laser');
    addLog('emission', `Starting simulation with ${lasers.length} laser source(s)`);

    lasers.forEach(laser => {
      let currentPos = { x: laser.x + 30, y: laser.y };
      let currentState = laser.properties.state || 'H';
      let intensity = 1.0;
      let angle = laser.rotation;
      
      const stateNames: Record<PolarizationState, string> = {
        H: 'Horizontal',
        V: 'Vertical',
        D: 'Diagonal (+45°)',
        A: 'Anti-diagonal (-45°)',
        R: 'Right circular',
        L: 'Left circular'
      };

      addLog('emission', `Laser "${laser.properties.label || laser.id}" emitting photon in ${stateNames[currentState]} polarization`, {
        componentId: laser.id,
        state: currentState,
        intensity: 1.0,
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
          newBeams.push({
            from: currentPos,
            to: { x: hitComponent.x, y: hitComponent.y },
            state: currentState,
            intensity,
          });

          if (hitComponent.type === 'polarizer') {
            const basis = hitComponent.properties.basis || 'rectilinear';
            const basisName = basis === 'rectilinear' ? 'Rectilinear (H/V)' : 'Diagonal (+45°/-45°)';
            
            if (basis === 'rectilinear' && (currentState === 'D' || currentState === 'A')) {
              const oldIntensity = intensity;
              intensity *= 0.5;
              addLog('interaction', `Polarizer "${hitComponent.properties.label || hitComponent.id}" (${basisName}) reduces diagonal photon intensity: ${oldIntensity.toFixed(2)} → ${intensity.toFixed(2)}`, {
                componentId: hitComponent.id,
                intensity,
              });
            } else if (basis === 'diagonal' && (currentState === 'H' || currentState === 'V')) {
              const oldIntensity = intensity;
              intensity *= 0.5;
              addLog('interaction', `Polarizer "${hitComponent.properties.label || hitComponent.id}" (${basisName}) reduces rectilinear photon intensity: ${oldIntensity.toFixed(2)} → ${intensity.toFixed(2)}`, {
                componentId: hitComponent.id,
                intensity,
              });
            } else {
              addLog('interaction', `Polarizer "${hitComponent.properties.label || hitComponent.id}" (${basisName}) passes photon (matching basis)`, {
                componentId: hitComponent.id,
                intensity,
              });
            }
            currentPos = { x: hitComponent.x + 30, y: hitComponent.y };
          } else if (hitComponent.type === 'beamsplitter') {
            const oldIntensity = intensity;
            intensity *= 0.5;
            addLog('interaction', `Beam splitter "${hitComponent.properties.label || hitComponent.id}" splits photon: ${oldIntensity.toFixed(2)} → ${intensity.toFixed(2)} (transmitted) + ${intensity.toFixed(2)} (reflected)`, {
              componentId: hitComponent.id,
              intensity,
            });
            
            const reflectedBeam = {
              from: { x: hitComponent.x, y: hitComponent.y },
              to: { x: hitComponent.x, y: hitComponent.y + 80 },
              state: currentState,
              intensity: intensity,
            };
            newBeams.push(reflectedBeam);
            currentPos = { x: hitComponent.x + 30, y: hitComponent.y };
          } else if (hitComponent.type === 'mirror') {
            angle = (angle + 90) % 360;
            addLog('interaction', `Mirror "${hitComponent.properties.label || hitComponent.id}" reflects photon by 90°`, {
              componentId: hitComponent.id,
              intensity,
            });
            currentPos = { x: hitComponent.x + 30, y: hitComponent.y };
          } else if (hitComponent.type === 'detector') {
            addLog('detection', `✓ Detector "${hitComponent.properties.label || hitComponent.id}" measured photon! State: ${stateNames[currentState]}, Intensity: ${intensity.toFixed(2)}`, {
              componentId: hitComponent.id,
              state: currentState,
              intensity,
            });
            break;
          } else {
            addLog('interaction', `Photon passed through "${hitComponent.properties.label || hitComponent.id}"`, {
              componentId: hitComponent.id,
              intensity,
            });
            currentPos = { x: hitComponent.x + 30, y: hitComponent.y };
          }
        } else {
          newBeams.push({
            from: currentPos,
            to: nextPos,
            state: currentState,
            intensity,
          });
          currentPos = nextPos;
        }

        if (nextPos.x < 0 || nextPos.x > 1000 || nextPos.y < 0 || nextPos.y > 800) {
          addLog('loss', `Photon exited the optical system (out of bounds)`, { intensity });
          break;
        }
      }
    });

    addLog('emission', `Simulation complete: ${logs.filter(l => l.type === 'detection').length} detection(s), ${logs.filter(l => l.type === 'interaction').length} interaction(s)`);

    setBeams(newBeams);
    setSimulationLogs(logs);
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
        onMouseDown={(e) => handleComponentMouseDown(e, component.id)}
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

  return (
    <div className="w-full h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900/80 to-slate-800/80 backdrop-blur-xl text-white p-5 flex items-center justify-between border-b border-slate-700/50 shadow-2xl">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
            <Zap size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-light tracking-wide">Quantum Optical Designer</h1>
            <p className="text-xs text-slate-400 font-light">Design & simulate quantum circuits</p>
          </div>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowInfo(!showInfo)}
            className="px-4 py-2 bg-slate-800/50 hover:bg-slate-700/50 rounded-lg flex items-center gap-2 transition-all border border-slate-700/50 backdrop-blur-sm text-sm font-light"
          >
            <Info size={14} />
            Info
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
            {isSimulating ? 'Simulating...' : 'Simulate'}
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
          <h3 className="font-light text-sm mb-2 text-blue-300">BB84 Quantum Key Distribution</h3>
          <p className="text-xs text-slate-300 mb-3 font-light leading-relaxed max-w-4xl">
            Quantum cryptography protocol where Alice transmits photons to Bob using random polarization bases. 
            Bob measures with randomly chosen bases—matching bases yield secure key bits. Eve's interception introduces detectable errors.
          </p>
          <div className="text-xs space-y-1 text-slate-400 font-light">
            <p><strong>Shortcuts:</strong> Delete/Backspace to remove | Ctrl+C to copy | Ctrl+V to paste | Click & drag to move</p>
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
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
              </div>
            ))}
          </div>

          {selectedComponent && (
            <div className="mt-8 pt-5 border-t border-slate-800/50">
              <h3 className="text-slate-300 font-light text-sm mb-3 tracking-wide">SELECTED</h3>
              <div className="space-y-2">
                <button
                  onClick={rotateComponent}
                  className="w-full px-4 py-2.5 bg-slate-800/50 hover:bg-slate-700/50 text-slate-200 rounded-lg flex items-center justify-center gap-2 transition-all border border-slate-700/50 text-sm font-light"
                >
                  <RotateCw size={14} />
                  Rotate 45°
                </button>
                <button
                  onClick={duplicateComponent}
                  className="w-full px-4 py-2.5 bg-blue-950/30 hover:bg-blue-900/40 text-blue-300 rounded-lg flex items-center justify-center gap-2 transition-all border border-blue-800/30 text-sm font-light"
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
          onClick={() => setSelectedComponent(null)}
        >
          {/* Subtle grid */}
          <div 
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage: 'radial-gradient(circle, #60a5fa 1px, transparent 1px)',
              backgroundSize: '40px 40px'
            }}
          />

          {/* Beams with glow effect */}
          <svg className="absolute inset-0 pointer-events-none" style={{ width: '100%', height: '100%' }}>
            <defs>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>
            {beams.map((beam, i) => (
              <g key={i}>
                <line
                  x1={beam.from.x}
                  y1={beam.from.y}
                  x2={beam.to.x}
                  y2={beam.to.y}
                  stroke={beam.state === 'H' || beam.state === 'V' ? '#ef4444' : '#a78bfa'}
                  strokeWidth="2"
                  opacity={beam.intensity * 0.8}
                  filter="url(#glow)"
                  className="animate-pulse"
                />
                <circle
                  cx={beam.to.x}
                  cy={beam.to.y}
                  r="3"
                  fill={beam.state === 'H' || beam.state === 'V' ? '#ef4444' : '#a78bfa'}
                  opacity={beam.intensity}
                  filter="url(#glow)"
                />
              </g>
            ))}
          </svg>

          {/* Components */}
          {components.map(renderComponent)}

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

        {/* Simulation Output Panel */}
        {showLogs && (
          <div className="w-96 bg-gradient-to-b from-slate-900/50 to-slate-900/30 backdrop-blur-xl border-l border-slate-800/50 flex flex-col">
            <div className="p-5 border-b border-slate-800/50 flex items-center justify-between">
              <h2 className="text-slate-300 font-light text-sm tracking-wide">SIMULATION OUTPUT</h2>
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
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuantumOpticsDesigner;