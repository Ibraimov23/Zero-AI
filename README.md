# Zero AI

Zero AI is an autonomous, browser-based voice assistant designed for interactive language learning and personal knowledge management. It operates directly on the user's device using Edge AI technologies, minimizing server dependency and eliminating subscription costs.

## Architecture and Core Concept

The core philosophy of Zero AI is "Zero Lag." To achieve a seamless, real-time conversational experience without UI freezing, the application architecture is strictly divided into three isolated layers:

1. **Main Thread (UI Layer):** 
   Handles only the visual rendering, including the reactive holographic sphere, background particles, and audio capture. It uses Web Audio API and Voice Activity Detection (VAD) to provide a completely hands-free experience.

2. **GPU Worker (AI Core):**
   An isolated background thread running ONNX models directly on the device's GPU (via WebGPU) or CPU (via WASM). It handles heavy audio processing tasks:
   - **Speech-to-Text (STT):** Uses a local `Whisper-tiny` model to transcribe user speech instantly.
   - **Text-to-Speech (TTS):** Uses a local `Kokoro-82M` model to synthesize a natural human voice from text.

3. **LLM Worker (Logic Layer):**
   Coordinates network requests and dialogue logic. It connects to the Gemini 2.5 Flash API via Server-Sent Events (SSE). It features a custom "Sentence Splitter" that intercepts the incoming text stream, cuts it into logical sentences, and immediately forwards them to the GPU Worker for voice synthesis, ensuring the AI starts speaking before the full text is even generated.

## How the Hands-Free System Works

The interaction feels natural, similar to speaking with a real person:

1. **Activation:** The user taps the microphone button once to start the session.
2. **Listening:** The system continuously captures audio.
3. **Voice Activity Detection (VAD):** An algorithm analyzes the audio frequencies in real-time. When the user stops speaking for 800 milliseconds, the system automatically cuts the recording and sends it for transcription.
4. **Real-time Synthesis:** As the AI thinks and streams text, the voice synthesis happens in parallel.
5. **Continuous Loop:** Once the AI finishes speaking its response, the microphone automatically reopens to listen to the user's next phrase.

## Tech Stack

- **Frontend:** React, TypeScript, Vite
- **Machine Learning (In-Browser):** Hugging Face Transformers.js (v3.2.0), ONNX Runtime Web
- **Voice Models:** Whisper-tiny.en (Local STT), Kokoro-82M-v1.0 (Local TTS)
- **Language Model:** Gemini 2.5 Flash API (Streaming)
- **Audio Processing:** Web Audio API, OfflineAudioContext, MediaRecorder

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ibraimov23/Zero-AI.git
   cd Zero-AI/web-app
   ```

2. Install dependencies:
   ```bash
   npm install
   ```
   *(Note: The project requires `kokoro-js` for TTS processing).*

3. Configure Environment Variables:
   Create a `.env` file in the `web-app` directory and add your Gemini API key:
   ```env
   VITE_GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. Download Local Models:
   The application requires the ONNX weights for Whisper and Kokoro to be placed in the `web-app/public/models/` directory. Due to their size, they are not included in the repository by default. You must download them via Git LFS from Hugging Face:
   - `onnx-community/whisper-tiny.en`
   - `onnx-community/Kokoro-82M-v1.0-ONNX`

5. Start the development server:
   ```bash
   npm run dev
   ```

## Future Roadmap (v2 & v3)

- Integration with a local embedding model (`bge-small-en-v1.5`) for offline vector search and "Second Brain" capabilities.
- Supabase integration for secure OAuth 2.0 authentication and user profile storage.
- Automated generation of personalized daily language assignments based on conversation history.
