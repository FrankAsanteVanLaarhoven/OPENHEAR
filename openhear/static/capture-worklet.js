/* Mono PCM accumulator. WebAudio-only; nothing is transmitted by this worklet. */
class OpenHearCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.position = 0;
  }
  process(inputs) {
    const mono = inputs[0] && inputs[0][0];
    if (!mono) return true;
    for (let i = 0; i < mono.length; i++) {
      this.buffer[this.position++] = mono[i];
      if (this.position === this.buffer.length) {
        this.port.postMessage(this.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(2048);
        this.position = 0;
      }
    }
    return true;
  }
}
registerProcessor('openhear-capture', OpenHearCaptureProcessor);
