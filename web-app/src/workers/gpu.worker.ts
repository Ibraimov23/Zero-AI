// gpu.worker.ts - Edge AI layer (WebGPU)
// This worker will load and run ONNX models: Whisper-tiny, Kokoro-82M, bge-small-en-v1.5

import {
  pipeline,
  env,
  AutomaticSpeechRecognitionPipeline,
} from '@huggingface/transformers';

// ==========================================
// 1. Environment Setup
// ==========================================
// Configure environment for local/remote models
// To ensure it works flawlessly on Vercel and locally (without LFS issues), we use the HuggingFace CDN
env.allowLocalModels = false;
env.allowRemoteModels = true;

// Fallback threads for WASM backend if WebGPU is blocked
if (env.backends.onnx.wasm) {
  env.backends.onnx.wasm.numThreads = 4;
}

// ==========================================
// 2. State Management
// ==========================================
interface ModelsState {
  speechToText: AutomaticSpeechRecognitionPipeline | null;
  embeddings: any | null;
}

const state: ModelsState = {
  speechToText: null,
  embeddings: null,
};

// ==========================================
// 3. Init Models
// ==========================================
async function initModels() {
  try {
    self.postMessage({ type: 'STATUS', payload: 'Checking WebGPU support...' });
    
    // Check if WebGPU is supported on this device/browser
    // WebGPU is currently unstable on many mobile Android/iOS devices for heavy transformer graphs
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const hasWebGPU = !isMobile && 'gpu' in navigator;
    
    const device = hasWebGPU ? 'webgpu' : 'wasm';
    const dtype = hasWebGPU ? 'fp32' : 'q4'; // Use quantized q4 on mobile CPU to save RAM!

    self.postMessage({
      type: 'STATUS',
      payload: `Initializing Whisper model (Device: ${device}, Dtype: ${dtype})...`,
    });

    // Initialize Whisper Pipeline from HuggingFace directly
    // Using fp32 or q4 for Whisper as some WebGPU implementations struggle with fp16 outer scope logits
    state.speechToText = await pipeline(
      'automatic-speech-recognition',
      'onnx-community/whisper-tiny.en', 
      {
        device: device as any,
        dtype: {
          encoder_model: dtype,
          decoder_model_merged: dtype,
        },
      }
    );

    // Later v2/v3: Initialize BGE here...

    self.postMessage({
      type: 'READY',
      payload: { device, dtype },
    });
  } catch (error) {
    console.error('Model initialization error:', error);
    self.postMessage({
      type: 'ERROR',
      payload: error instanceof Error ? error.message : String(error),
    });
  }
}

// ==========================================
// 4. Transcribe Audio
// ==========================================
async function transcribeAudio(audioData: Float32Array) {
  if (!state.speechToText) {
    self.postMessage({
      type: 'ERROR',
      payload: 'Speech-to-Text model is not initialized yet.',
    });
    return;
  }

  try {
    self.postMessage({ type: 'STATUS', payload: 'Transcribing audio...' });

    const result = await state.speechToText(audioData, {
      chunk_length_s: 30,
      stride_length_s: 5,
      // Removed `language` and `task` because whisper-tiny.en is an English-only model
    });

    self.postMessage({ type: 'TRANSCRIPT_SUCCESS', payload: result.text });
  } catch (error) {
    console.error('Transcription error:', error);
    self.postMessage({
      type: 'ERROR',
      payload: error instanceof Error ? error.message : String(error),
    });
  }
}

// ==========================================
// 5. Message Event Listener
// ==========================================
self.addEventListener('message', async (event: MessageEvent) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'INIT_MODELS':
      await initModels();
      break;

    case 'PROCESS_AUDIO':
      if (payload instanceof ArrayBuffer) {
        // Zero-copy conversion if possible, or simple view wrapping
        const audioArray = new Float32Array(payload);
        await transcribeAudio(audioArray);
      } else if (payload instanceof Float32Array) {
        await transcribeAudio(payload);
      } else {
        self.postMessage({
          type: 'ERROR',
          payload: 'Invalid audio format. Expected ArrayBuffer or Float32Array.',
        });
      }
      break;

    default:
      console.warn('Unknown message type received in gpu.worker:', type);
  }
});
