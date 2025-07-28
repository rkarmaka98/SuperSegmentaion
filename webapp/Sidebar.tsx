import React, { useEffect, useState } from 'react';
import { getLayer, setLayer, LayerName, addLayerListener, removeLayerListener } from './LayerStore';

// map to render label for each checkbox
const layers: Record<LayerName, string> = {
  segmentation: 'Segmentation',
  keypoints: 'Keypoints',
  matching: 'Matching',
  confidence: 'Confidence',
};

export const Sidebar: React.FC = () => {
  // local state mirrors the store state so UI stays reactive
  const [visibility, setVisibility] = useState(() => {
    const state: Record<LayerName, boolean> = {
      segmentation: getLayer('segmentation'),
      keypoints: getLayer('keypoints'),
      matching: getLayer('matching'),
      confidence: getLayer('confidence'),
    };
    return state;
  });

  useEffect(() => {
    // update component when store changes via external actions
    const listener = (evt: Event) => {
      const detail = (evt as CustomEvent).detail as { layer: LayerName; visible: boolean };
      setVisibility(prev => ({ ...prev, [detail.layer]: detail.visible }));
    };
    addLayerListener(listener);
    // cleanup when unmounting
    return () => removeLayerListener(listener);
  }, []);

  const onToggle = (layer: LayerName) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.checked;
    // update store so other views are notified
    setLayer(layer, value);
  };

  return (
    <div className="sidebar">
      {Object.entries(layers).map(([name, label]) => (
        <label key={name} style={{ display: 'block' }}>
          <input
            type="checkbox"
            checked={visibility[name as LayerName]}
            onChange={onToggle(name as LayerName)}
          />
          {label}
        </label>
      ))}
    </div>
  );
};
