import { useEffect, useRef, useState, useMemo } from 'react';
import './App.css';

function App() {
  const [status, setStatus] = useState<string>('Initializing...');
  const [isReady, setIsReady] = useState(false);
  const [transcript, setTranscript] = useState<string>('');
  
  const [isSessionActive, setIsSessionActive] = useState(false); // Master toggle for hands-free
  const [isListening, setIsListening] = useState(false); // When actively collecting audio
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [aiResponse, setAiResponse] = useState<string>('');
  
  const gpuWorkerRef = useRef<Worker | null>(null);
  const llmWorkerRef = useRef<Worker | null>(null);

  // OpenAI TTS integration
  const synthesizeWithOpenAI = async (text: string) => {
    if (!text.trim()) return;
    try {
      const response = await fetch('https://api.openai.com/v1/audio/speech', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'tts-1', // You can use tts-1-hd for higher quality
          input: text,
          voice: 'nova', // Alloy, echo, fable, onyx, nova, or shimmer
          response_format: 'mp3',
        }),
      });

      if (!response.ok) throw new Error(`OpenAI TTS Error: ${response.statusText}`);
      
      const arrayBuffer = await response.arrayBuffer();
      audioQueueRef.current.push(arrayBuffer);
      playNextAudio();
    } catch (error) {
      console.error('TTS Error:', error);
    }
  };

  // Audio State Refs
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // VAD (Voice Activity Detection) Refs
  const vadContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const vadFrameRef = useRef<number | null>(null);
  const hasSpokenRef = useRef<boolean>(false);
  const silenceStartRef = useRef<number | null>(null);
  const shouldProcessAudioRef = useRef<boolean>(false); // to prevent processing on forced stop

  // Keep a mutable ref of session active state for use in callbacks
  const isSessionActiveRef = useRef<boolean>(false);

  // TTS Playback Queue Refs
  const audioQueueRef = useRef<ArrayBuffer[]>([]);
  const isPlayingRef = useRef<boolean>(false);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const currentAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);

  // Interrupt AI playback and generation
  const interruptAi = () => {
    audioQueueRef.current = []; // Clear queue
    if (currentAudioSourceRef.current) {
      try {
        currentAudioSourceRef.current.stop();
      } catch (e) {
        // Ignore if already stopped
      }
      currentAudioSourceRef.current = null;
    }
    isPlayingRef.current = false;
    setIsAiSpeaking(false);
    llmWorkerRef.current?.postMessage({ type: 'ABORT_GENERATION' });
  };

  // Helper to play TTS audio sequentially
  const playNextAudio = async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      if (audioQueueRef.current.length === 0 && !isPlayingRef.current) {
        setIsAiSpeaking(false);
        // Automatically restart listening if session is still active!
        if (isSessionActiveRef.current && !isListening) {
          setTimeout(() => {
            if (isSessionActiveRef.current) startListening();
          }, 600); // Short pause before mic re-opens
        }
      }
      return;
    }

    isPlayingRef.current = true;
    setIsAiSpeaking(true);
    const audioData = audioQueueRef.current.shift();
    if (!audioData) {
      isPlayingRef.current = false;
      setIsAiSpeaking(false);
      return;
    }

    if (!playbackContextRef.current) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      playbackContextRef.current = new AudioContextClass(); // Use default sample rate for decoding
    }

    const audioCtx = playbackContextRef.current;
    if (audioCtx.state === 'suspended') audioCtx.resume();

    try {
      // Decode the MP3 array buffer from OpenAI
      const audioBuffer = await audioCtx.decodeAudioData(audioData);
      
      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);
      currentAudioSourceRef.current = source;
      
      source.onended = () => {
        currentAudioSourceRef.current = null;
        isPlayingRef.current = false;
        if (audioQueueRef.current.length === 0) {
          setIsAiSpeaking(false);
          // Automatically restart listening if session is still active!
          if (isSessionActiveRef.current) {
            setTimeout(() => {
              if (isSessionActiveRef.current) startListening();
            }, 600);
          }
        } else {
          playNextAudio();
        }
      };

      source.start(0);
    } catch (error) {
      console.error('Audio playback error:', error);
      isPlayingRef.current = false;
      playNextAudio(); // Skip to next if decoding fails
    }
  };

  useEffect(() => {
    // Initialize Web Workers as ECMAScript modules
    gpuWorkerRef.current = new Worker(new URL('./workers/gpu.worker.ts', import.meta.url), { type: 'module' });
    llmWorkerRef.current = new Worker(new URL('./workers/llm.worker.ts', import.meta.url), { type: 'module' });

    // Handle messages from GPU Worker
    gpuWorkerRef.current.onmessage = (e: MessageEvent) => {
      const { type, payload } = e.data;
      
      switch (type) {
        case 'STATUS':
          console.log('[GPU Worker Status]:', payload);
          setStatus(payload);
          break;
          
        case 'READY':
          console.log('[GPU Worker Ready]:', payload);
          setStatus(`Ready (Zero Lag UI) - Device: ${payload.device}`);
          setIsReady(true);
          break;
          
        case 'TRANSCRIPT_SUCCESS':
          console.log('[GPU Worker Transcript]:', payload);
          setTranscript(payload);
          setStatus('Reasoning (Gemini 2.5 Flash)...');
          setAiResponse(''); // Clear previous AI response
          // Forward to LLM worker
          llmWorkerRef.current?.postMessage({ type: 'GENERATE_RESPONSE', payload: { prompt: payload } });
          break;
          
        case 'AUDIO_CHUNK_READY':
          // Legacy support if Kokoro is still used
          if (payload instanceof ArrayBuffer) {
            // We ignore local Kokoro audio now to prevent duplicate playback
          }
          break;
          
        case 'ERROR':
          console.error('[GPU Worker Error]:', payload);
          setStatus(`Error: ${payload}`);
          break;
          
        default:
          console.log('Unknown message from GPU Worker:', e.data);
      }
    };

    // Handle messages from LLM Worker
    llmWorkerRef.current.onmessage = (e: MessageEvent) => {
      const { type, payload } = e.data;
      
      switch (type) {
        case 'LLM_READY':
          console.log('[LLM Worker Ready]');
          break;
          
        case 'TEXT_CHUNK':
          setAiResponse((prev) => prev + payload);
          break;
          
        case 'SENTENCE_READY':
          console.log('[LLM Sentence Ready for TTS]:', payload);
          // 🛑 2. Use Cloud TTS (OpenAI) instead of local Kokoro for ultra-low latency
          synthesizeWithOpenAI(payload);
          break;
          
        case 'STREAM_END':
          console.log('[LLM Stream End]');
          setStatus('Ready (Zero Lag UI) - Waiting for next interaction');
          break;
          
        case 'ERROR':
          console.error('[LLM Worker Error]:', payload);
          setStatus(`LLM Error: ${payload}`);
          break;
          
        default:
          console.log('Unknown message from LLM Worker:', e.data);
      }
    };

    // Send the starting signal to build the local AI Core
    gpuWorkerRef.current.postMessage({ type: 'INIT_MODELS' });
    llmWorkerRef.current.postMessage({ type: 'INIT_LLM' });

    // Cleanup workers on unmount
    return () => {
      if (vadFrameRef.current) cancelAnimationFrame(vadFrameRef.current);
      if (vadContextRef.current) vadContextRef.current.close();
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      gpuWorkerRef.current?.terminate();
      llmWorkerRef.current?.terminate();
    };
  }, []);

  const toggleSession = async () => {
    if (isSessionActive) {
      // Stop completely
      setIsSessionActive(false);
      isSessionActiveRef.current = false;
      stopListening(false);
      setStatus('Session ended');
    } else {
      // Start continuous session
      setIsSessionActive(true);
      isSessionActiveRef.current = true;
      await startListening();
    }
  };

  const startListening = async () => {
    if (!isSessionActiveRef.current) return;

    interruptAi(); // STOP AI IMMEDIATELY WHEN LISTENING STARTS

    try {
      setTranscript(''); // Clear previous transcript
      setAiResponse('');
      setStatus('Listening...');
      setIsListening(true);
      hasSpokenRef.current = false;
      silenceStartRef.current = null;
      shouldProcessAudioRef.current = true;
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }, 
        video: false 
      });
      mediaStreamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        setIsListening(false);

        if (!shouldProcessAudioRef.current) {
          return; // Discard audio if stopped forcefully
        }

        setStatus('Processing audio (Resampling)...');
        
        try {
          const blob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType });
          const arrayBuffer = await blob.arrayBuffer();
          
          // Initialize AudioContext inside user gesture
          const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
          const audioContext = new AudioContextClass();
          
          const decodedAudio = await audioContext.decodeAudioData(arrayBuffer);

          // Whisper requires 16000Hz mono audio
          const offlineContext = new OfflineAudioContext(
            1, // Mono channel
            decodedAudio.duration * 16000, // Total frames
            16000 // Sample rate
          );
          
          const source = offlineContext.createBufferSource();
          source.buffer = decodedAudio;
          source.connect(offlineContext.destination);
          source.start(0);

          const renderedBuffer = await offlineContext.startRendering();
          const float32Array = renderedBuffer.getChannelData(0); // Get mono channel
          
          // Send to GPU Worker via Transferable Objects
          const buffer = float32Array.buffer;
          setStatus('Sending to AI Core for transcription...');
          gpuWorkerRef.current?.postMessage(
            { type: 'PROCESS_AUDIO', payload: buffer },
            [buffer]
          );
          
          // Cleanup
          audioContext.close();
        } catch (error) {
          console.error('Audio processing error:', error);
          setStatus('Error: Failed to process audio');
          if (isSessionActiveRef.current) startListening();
        }
      };

      // VAD setup
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      vadContextRef.current = new AudioContextClass();
      const source = vadContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = vadContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 512;
      source.connect(analyserRef.current);

      const checkSilence = () => {
        if (!analyserRef.current || !isSessionActiveRef.current) return;
        
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        
        const sum = dataArray.reduce((a, b) => a + b, 0);
        const avg = sum / dataArray.length;

        // VAD Logic
        if (avg > 10) { 
          // Volume threshold exceeded (User is speaking)
          hasSpokenRef.current = true;
          silenceStartRef.current = null;
        } else if (hasSpokenRef.current) {
          // User has spoken, now detecting silence
          if (!silenceStartRef.current) {
            silenceStartRef.current = Date.now();
          } else if (Date.now() - silenceStartRef.current > 800) { 
            // 800ms of silence detected (Fast Mobile Response!)
            console.log('Silence detected! Stopping mic to process...');
            stopListening(true);
            return;
          }
        }
        vadFrameRef.current = requestAnimationFrame(checkSilence);
      };

      mediaRecorder.start();
      checkSilence();
      
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setStatus('Error: Could not access microphone');
      setIsSessionActive(false);
      isSessionActiveRef.current = false;
    }
  };

  const stopListening = (process: boolean) => {
    shouldProcessAudioRef.current = process;
    
    if (vadFrameRef.current) cancelAnimationFrame(vadFrameRef.current);
    if (vadContextRef.current) {
      vadContextRef.current.close();
      vadContextRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    setIsListening(false);
  };

  // Generate random particles for background
  const particles = useMemo(() => {
    return Array.from({ length: 25 }).map((_, i) => ({
      id: i,
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      delay: `${Math.random() * 5}s`,
      duration: `${10 + Math.random() * 15}s`,
      size: `${1 + Math.random() * 3}px`
    }));
  }, []);

  return (
    <div className="app-container">
      {/* Background Particles */}
      <div className="particles-layer">
        {particles.map(p => (
          <div 
            key={p.id} 
            className="particle" 
            style={{ 
              left: p.left, 
              top: p.top, 
              animationDelay: p.delay, 
              animationDuration: p.duration,
              width: p.size,
              height: p.size
            }}
          />
        ))}
      </div>

      {/* Greeting and Status Header */}
      <div className="status-header">
        <div className="greeting-name">Zero AI by Nursultan and Aliya</div>
        <div className="main-prompt">
          {isSessionActive ? (isListening ? "I'M LISTENING" : isAiSpeaking ? "ZERO AI" : "THINKING...") : "SAY SOMETHING"}
        </div>
        <div className="status-text">{status}</div>
      </div>
      
      <div className={`orb-container ${isListening ? 'active' : ''} ${isAiSpeaking ? 'ai-speaking' : ''}`}>
        <div className="orb">
          <div className="petal petal-1"></div>
          <div className="petal petal-2"></div>
          <div className="petal petal-3"></div>
          <div className="petal petal-4"></div>
          <div className="orb-inner"></div>
          <div className="orb-core"></div>
        </div>
      </div>

      <div className="text-container">
        {transcript && !isListening && (
          <div className="user-text" style={{ color: '#94a3b8', marginBottom: '1rem', fontSize: '0.9rem' }}>
            "{transcript}"
          </div>
        )}
        <div className="ai-text">
          {aiResponse ? aiResponse : <span className="placeholder-text">Wait for response...</span>}
        </div>
      </div>

      <div className="bottom-controls">
        <div className="mic-wrapper">
          {isListening && (
            <>
              <div className="ripple ripple-1"></div>
              <div className="ripple ripple-2"></div>
              <div className="ripple ripple-3"></div>
            </>
          )}
          <button 
            className={`mic-button ${isSessionActive ? 'recording' : ''}`}
            onClick={toggleSession}
            disabled={!isReady && !isSessionActive}
          >
            {isSessionActive ? (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
                <rect x="7" y="7" width="10" height="10" rx="2" />
              </svg>
            ) : (
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="22"></line>
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
