export type LayerName = 'segmentation' | 'keypoints' | 'matching' | 'confidence';

// State object holds the visibility of each layer in a single location
const state: Record<LayerName, boolean> = {
  segmentation: true,
  keypoints: true,
  matching: true,
  confidence: true,
};

// EventTarget allows other modules to listen for changes
const emitter = new EventTarget();

export function getLayer(name: LayerName) {
  return state[name];
}

export function setLayer(name: LayerName, value: boolean) {
  state[name] = value;
  // notify listeners that a layer visibility changed
  emitter.dispatchEvent(
    new CustomEvent('layerChange', { detail: { layer: name, visible: value } })
  );
}

export function addLayerListener(listener: (evt: CustomEvent) => void) {
  // addEventListener typed as EventListener, cast to handle CustomEvent
  emitter.addEventListener('layerChange', listener as EventListener);
}

export function removeLayerListener(listener: (evt: CustomEvent) => void) {
  emitter.removeEventListener('layerChange', listener as EventListener);
}
