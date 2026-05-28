import { useEffect, useRef, useState, useMemo, type CSSProperties } from 'react';
import './App.css';

type TutorMode = 'grammar' | 'free_speaking';

type BrowserSpeechRecognition = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  maxAlternatives: number;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type BrowserSpeechRecognitionConstructor = new () => BrowserSpeechRecognition;

declare global {
  interface Window {
    SpeechRecognition?: BrowserSpeechRecognitionConstructor;
    webkitSpeechRecognition?: BrowserSpeechRecognitionConstructor;
  }
}

const USER_END_OF_TURN_MS = 2200;
const AI_TO_USER_RESUME_MS = 2000;
const PRE_SPEECH_CUE_MS = 180;
const SOFT_BARGE_IN_CUE_MS = 240;
const VOICE_BARGE_IN_TRIGGER_MS = 560;
const VOICE_BARGE_IN_RELEASE_MS = 180;
const VOICE_BARGE_IN_COOLDOWN_MS = 1600;
const VOICE_BARGE_IN_VOLUME_THRESHOLD = 24;
const SHORT_VALID_UTTERANCE_MIN_MS = 420;
const SUSTAINED_SPEECH_MIN_MS = 750;
const SHORT_VALID_UTTERANCES = new Set([
  'yes',
  'no',
  'okay',
  'ok',
  'hello',
  'hi',
  'thanks',
  'sorry',
  'sure',
  'maybe',
]);
const NOISE_ONLY_UTTERANCES = new Set([
  'uh',
  'um',
  'hmm',
  'mm',
  'ah',
  'oh',
  'eh',
]);

function getMeaningfulTokens(text: string) {
  return text.toLowerCase().match(/[a-z]+(?:'[a-z]+)?/g) ?? [];
}

function isLikelyNoiseTranscript(text: string, speechDurationMs: number) {
  const normalized = text.trim().toLowerCase();
  const tokens = getMeaningfulTokens(normalized);

  if (!normalized || tokens.length === 0) return true;
  if (tokens.every(token => NOISE_ONLY_UTTERANCES.has(token))) return true;

  if (tokens.length === 1) {
    const [token] = tokens;
    if (SHORT_VALID_UTTERANCES.has(token)) {
      return speechDurationMs < SHORT_VALID_UTTERANCE_MIN_MS;
    }

    if (token.length <= 2) return true;
    if (speechDurationMs < SUSTAINED_SPEECH_MIN_MS && token.length < 5) return true;
  }

  return false;
}

interface SessionMetrics {
  tutorMode: TutorMode;
  lessonFocus: string;
  estimatedInputTokens: number;
  estimatedOutputTokens: number;
  estimatedTotalTokens: number;
  tokenBudget: number;
  memoryItems: number;
}

type VoiceTelemetryStatus = 'completed' | 'retryable_error' | 'error' | 'aborted' | 'ignored_noise';

interface VoiceTurnTelemetry {
  turnId: number;
  startedAt: number;
  sttMode: 'webrtc' | 'browser' | 'cloud';
  realtimeFailureReason: string | null;
  firstTranscriptAt: number | null;
  llmRequestedAt: number | null;
  firstLlmChunkAt: number | null;
  firstAssistantOutputAt: number | null;
  speechDurationMs: number | null;
  transcriptChars: number;
  aiChars: number;
  ttsFallbackCount: number;
  status: VoiceTelemetryStatus | null;
  error: string | null;
}

const DEFAULT_SESSION_METRICS: SessionMetrics = {
  tutorMode: 'free_speaking',
  lessonFocus: 'general conversation',
  estimatedInputTokens: 0,
  estimatedOutputTokens: 0,
  estimatedTotalTokens: 0,
  tokenBudget: 1800,
  memoryItems: 0,
};

const TUTOR_MODE_OPTIONS: Array<{ value: TutorMode; label: string; hint: string }> = [
  { value: 'grammar', label: 'Grammar', hint: 'Fix grammar and explain clearly' },
  { value: 'free_speaking', label: 'Free Speaking', hint: 'Natural conversation practice' },
];

const LESSON_FOCUS_OPTIONS: Record<TutorMode, string[]> = {
  grammar: ['past tense', 'articles', 'prepositions', 'sentence order'],
  free_speaking: ['general conversation'],
};

function App() {
  const [status, setStatus] = useState<string>('Initializing...');
  const [isReady, setIsReady] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [transcript, setTranscript] = useState<string>('');
  
  const [isListening, setIsListening] = useState(false);
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [isPreSpeechCue, setIsPreSpeechCue] = useState(false);
  const [isBargeInIntent, setIsBargeInIntent] = useState(false);
  const [aiResponse, setAiResponse] = useState<string>('');
  const [liveAiPreview, setLiveAiPreview] = useState<string>('');
  const [tutorMode, setTutorMode] = useState<TutorMode>('free_speaking');
  const [sessionMetrics, setSessionMetrics] = useState<SessionMetrics>(DEFAULT_SESSION_METRICS);
  
  const llmWorkerRef = useRef<Worker | null>(null);

  const isListeningRef = useRef(false);
  const isAiSpeakingRef = useRef(false);
  const isThinkingRef = useRef(false);
  const realtimePeerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const realtimeDataChannelRef = useRef<RTCDataChannel | null>(null);
  const realtimeTranscriptByItemIdRef = useRef<Map<string, string>>(new Map());
  const realtimeModeRef = useRef<'webrtc' | 'browser' | 'cloud'>('cloud');
  const handledListeningTurnRef = useRef(false);
  const telemetryTurnIdRef = useRef(0);
  const turnTelemetryRef = useRef<VoiceTurnTelemetry | null>(null);

  const closeRealtimeTranscriptionSession = () => {
    if (realtimeDataChannelRef.current) {
      try {
        realtimeDataChannelRef.current.close();
      } catch {
        // noop
      }
      realtimeDataChannelRef.current = null;
    }

    if (realtimePeerConnectionRef.current) {
      try {
        realtimePeerConnectionRef.current.close();
      } catch {
        // noop
      }
      realtimePeerConnectionRef.current = null;
    }

    realtimeTranscriptByItemIdRef.current.clear();
    if (realtimeModeRef.current === 'webrtc') {
      realtimeModeRef.current = 'cloud';
    }
  };

  const getSpeechRecognitionConstructor = () =>
    typeof window !== 'undefined' ? window.SpeechRecognition || window.webkitSpeechRecognition : undefined;

  const beginTurnTelemetry = () => {
    telemetryTurnIdRef.current += 1;
    turnTelemetryRef.current = {
      turnId: telemetryTurnIdRef.current,
      startedAt: Date.now(),
      sttMode: 'cloud',
      realtimeFailureReason: null,
      firstTranscriptAt: null,
      llmRequestedAt: null,
      firstLlmChunkAt: null,
      firstAssistantOutputAt: null,
      speechDurationMs: null,
      transcriptChars: 0,
      aiChars: 0,
      ttsFallbackCount: 0,
      status: null,
      error: null,
    };
  };

  const patchTurnTelemetry = (patch: Partial<VoiceTurnTelemetry>) => {
    if (!turnTelemetryRef.current) return;
    turnTelemetryRef.current = {
      ...turnTelemetryRef.current,
      ...patch,
    };
  };

  const setTurnSttMode = (mode: 'webrtc' | 'browser' | 'cloud') => {
    patchTurnTelemetry({ sttMode: mode });
  };

  const setTurnRealtimeFailureReason = (reason: string) => {
    patchTurnTelemetry({ realtimeFailureReason: reason });
  };

  const markTranscriptVisible = (text: string, speechDurationMs?: number) => {
    const current = turnTelemetryRef.current;
    const nextText = text.trim();
    if (!current || !nextText) return;

    patchTurnTelemetry({
      firstTranscriptAt: current.firstTranscriptAt ?? Date.now(),
      transcriptChars: Math.max(current.transcriptChars, nextText.length),
      speechDurationMs: speechDurationMs ?? current.speechDurationMs,
    });
  };

  const markLlmRequested = () => {
    const current = turnTelemetryRef.current;
    if (!current || current.llmRequestedAt) return;
    patchTurnTelemetry({ llmRequestedAt: Date.now() });
  };

  const markLlmChunk = (chunk: string) => {
    const current = turnTelemetryRef.current;
    if (!current || !chunk) return;

    patchTurnTelemetry({
      firstLlmChunkAt: current.firstLlmChunkAt ?? Date.now(),
      aiChars: current.aiChars + chunk.length,
    });
  };

  const markAssistantOutput = (textOnly = false) => {
    const current = turnTelemetryRef.current;
    if (!current) return;

    patchTurnTelemetry({
      firstAssistantOutputAt: current.firstAssistantOutputAt ?? Date.now(),
      ttsFallbackCount: current.ttsFallbackCount + (textOnly ? 1 : 0),
    });
  };

  const finalizeTurnTelemetry = (status: VoiceTelemetryStatus, overrides?: Partial<VoiceTurnTelemetry>) => {
    const current = turnTelemetryRef.current;
    if (!current || current.status) return;

    const finalTelemetry = {
      ...current,
      ...overrides,
      status,
    };
    const elapsed = (timestamp: number | null) => (timestamp ? timestamp - finalTelemetry.startedAt : null);

    console.info('[Voice Telemetry]', {
      turnId: finalTelemetry.turnId,
      status: finalTelemetry.status,
      sttMode: finalTelemetry.sttMode,
      realtimeFailureReason: finalTelemetry.realtimeFailureReason,
      speechDurationMs: finalTelemetry.speechDurationMs,
      firstTranscriptMs: elapsed(finalTelemetry.firstTranscriptAt),
      llmRequestMs: elapsed(finalTelemetry.llmRequestedAt),
      firstLlmChunkMs: elapsed(finalTelemetry.firstLlmChunkAt),
      firstAssistantOutputMs: elapsed(finalTelemetry.firstAssistantOutputAt),
      transcriptChars: finalTelemetry.transcriptChars,
      aiChars: finalTelemetry.aiChars,
      ttsFallbackCount: finalTelemetry.ttsFallbackCount,
      error: finalTelemetry.error,
    });

    turnTelemetryRef.current = null;
  };

  const submitUserPrompt = (rawText: string, speechDuration: number, source: 'Cloud STT' | 'Streaming STT') => {
    const text = rawText.trim();
    const lowerText = text.toLowerCase();
    const resolvedSttMode =
      source === 'Cloud STT'
        ? 'cloud'
        : realtimeModeRef.current === 'browser'
          ? 'browser'
          : 'webrtc';

    setTurnSttMode(resolvedSttMode);
    markTranscriptVisible(text, speechDuration);

    if (
      !text ||
      lowerText === 'thank you.' ||
      lowerText === 'thank you for watching.' ||
      lowerText === 'you' ||
      lowerText.includes('amara.org')
    ) {
      console.log(`[${source}] Ignored hallucination/silence:`, text);
      setStatus('Ready');
      setIsThinking(false);
      finalizeTurnTelemetry('ignored_noise', { error: 'Ignored hallucination or silence.' });
      if (isSessionActiveRef.current) {
        checkAiFinishedAndResume();
      }
      return false;
    }

    if (isLikelyNoiseTranscript(text, speechDuration)) {
      console.log(`[${source}] Ignored likely noise / filler:`, { text, speechDuration });
      setStatus('Ready');
      setIsThinking(false);
      finalizeTurnTelemetry('ignored_noise', { error: 'Ignored likely noise or filler.' });
      if (isSessionActiveRef.current) {
        checkAiFinishedAndResume();
      }
      return false;
    }

    console.log(`[${source} Transcript]:`, text);
    setIsThinking(true);
    activeResponseIdRef.current += 1;
    currentSentenceIndexRef.current = 0;
    nextSentenceToPlayRef.current = 0;
    pendingSentenceAudioRef.current.clear();
    pendingTtsRequestsRef.current = 0;
    audioQueueRef.current = [];
    setTranscript(text);
    setStatus('Reasoning (Gemini 2.5 Flash)...');
    setAiResponse('');
    setLiveAiPreview('');
    livePreviewResponseIdRef.current = activeResponseIdRef.current;
    hasStartedSpeechForResponseRef.current = false;
    isLlmStreamingRef.current = true;
    markLlmRequested();
    llmWorkerRef.current?.postMessage({
      type: 'GENERATE_RESPONSE',
      payload: { prompt: text, responseId: activeResponseIdRef.current }
    });
    return true;
  };

  // OpenAI Whisper STT Integration (Cloud - Ultra Fast)
  const transcribeWithOpenAI = async (blob: Blob) => {
    try {
      const formData = new FormData();
      formData.append('file', blob, 'audio.webm');
      formData.append('model', 'whisper-1');
      formData.append('language', 'en'); // Force English for speed
      formData.append('temperature', '0'); // Strict deterministic transcription to prevent hallucinations
      formData.append('prompt', 'Hello, this is a conversation. Please do not transcribe silence or background noise.');

      const response = await fetch('/api/stt', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`OpenAI STT Error: ${response.statusText}`);
      
      const data = await response.json();
      if (data.text) {
        const speechDuration = lastSpeechDurationRef.current;
        setTurnSttMode('cloud');
        submitUserPrompt(data.text, speechDuration, 'Cloud STT');
      }
    } catch (error) {
      console.error('Cloud STT Error:', error);
      setStatus('Error: Cloud STT Failed');
      setIsThinking(false);
      if (isSessionActiveRef.current) checkAiFinishedAndResume();
    }
  };

  const flushPendingSentenceAudio = () => {
    while (pendingSentenceAudioRef.current.has(nextSentenceToPlayRef.current)) {
      const nextAudio = pendingSentenceAudioRef.current.get(nextSentenceToPlayRef.current);
      pendingSentenceAudioRef.current.delete(nextSentenceToPlayRef.current);
      if (nextAudio) {
        audioQueueRef.current.push(nextAudio);
      }
      nextSentenceToPlayRef.current += 1;
    }
    playNextAudio();
  };

  // OpenAI TTS integration
  const synthesizeWithOpenAI = async (text: string, responseId: number, sentenceIndex: number) => {
    if (!text.trim()) return;
    if (responseId !== activeResponseIdRef.current) return;
    pendingTtsRequestsRef.current++;
    try {
      let arrayBuffer: ArrayBuffer | null = null;
      let lastError: unknown = null;

      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const response = await fetch('/api/tts', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
          });

          if (!response.ok) throw new Error(`OpenAI TTS Error: ${response.statusText}`);

          arrayBuffer = await response.arrayBuffer();
          break;
        } catch (error) {
          lastError = error;
          if (attempt === 0) {
            await new Promise(resolve => window.setTimeout(resolve, 150));
          }
        }
      }

      if (responseId !== activeResponseIdRef.current) return;

      if (!arrayBuffer) {
        console.error('TTS Error: Falling back to subtitles-only chunk', lastError);
        pendingSentenceAudioRef.current.set(sentenceIndex, { buffer: null, text, responseId, textOnly: true });
        flushPendingSentenceAudio();
        return;
      }

      // Keep sentence order stable even if TTS responses arrive out of order.
      pendingSentenceAudioRef.current.set(sentenceIndex, { buffer: arrayBuffer, text, responseId, textOnly: false });
      flushPendingSentenceAudio();
    } finally {
      if (responseId === activeResponseIdRef.current) {
        pendingTtsRequestsRef.current = Math.max(0, pendingTtsRequestsRef.current - 1);
        checkAiFinishedAndResume();
      }
    }
  };

  // Audio State Refs
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const passiveBargeInStreamRef = useRef<MediaStream | null>(null);
  const passiveBargeInContextRef = useRef<AudioContext | null>(null);
  const passiveBargeInAnalyserRef = useRef<AnalyserNode | null>(null);
  const passiveBargeInFrameRef = useRef<number | null>(null);
  const passiveBargeInSpeechStartRef = useRef<number | null>(null);
  const passiveBargeInSilenceStartRef = useRef<number | null>(null);
  const passiveBargeInLastTriggerRef = useRef<number>(0);
  const passiveBargeInStartingRef = useRef<boolean>(false);
  const softBargeInInProgressRef = useRef<boolean>(false);

  // VAD (Voice Activity Detection) Refs
  const vadContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const vadFrameRef = useRef<number | null>(null);
  const hasSpokenRef = useRef<boolean>(false);
  const silenceStartRef = useRef<number | null>(null);
  const speechStartRef = useRef<number | null>(null);
  const lastSpeechDurationRef = useRef<number>(0);
  const shouldProcessAudioRef = useRef<boolean>(false); // to prevent processing on forced stop
  const audioVolumeRef = useRef<number>(1); // To store current audio volume for visualizer

  // Utility: Trigger Haptic Feedback (vibration on mobile)
  const triggerHaptic = (type: 'light' | 'medium' | 'heavy' = 'light') => {
    if (typeof window !== 'undefined' && typeof window.navigator?.vibrate === 'function') {
      if (type === 'light') navigator.vibrate(10);
      else if (type === 'medium') navigator.vibrate(20);
      else navigator.vibrate([20, 30, 20]); // Double pulse
    }
  };

  // TTS Playback Queue Refs
  const audioQueueRef = useRef<{buffer: ArrayBuffer | null, text: string, responseId: number, textOnly: boolean}[]>([]);
  const isPlayingRef = useRef<boolean>(false);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const currentAudioSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const typewriterIntervalRef = useRef<number | null>(null);
  const resumeTimeoutRef = useRef<number | null>(null);
  const preSpeechCueTimeoutRef = useRef<number | null>(null);
  const bargeInIntentTimeoutRef = useRef<number | null>(null);
  const isSessionActiveRef = useRef<boolean>(false);
  const activeResponseIdRef = useRef<number>(0);
  const livePreviewResponseIdRef = useRef<number>(0);
  const hasStartedSpeechForResponseRef = useRef<boolean>(false);
  const currentSentenceIndexRef = useRef<number>(0);
  const nextSentenceToPlayRef = useRef<number>(0);
  const pendingSentenceAudioRef = useRef<Map<number, {buffer: ArrayBuffer | null, text: string, responseId: number, textOnly: boolean}>>(new Map());
  const speechRecognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const speechRecognitionModeRef = useRef<'streaming' | 'cloud'>('cloud');
  const speechFinalTranscriptRef = useRef<string>('');
  const speechInterimTranscriptRef = useRef<string>('');
  const recognitionStopRequestedRef = useRef<boolean>(false);
  const recognitionHasErrorRef = useRef<boolean>(false);
  const recognitionEndedRef = useRef<boolean>(true);
  const streamingTranscriptHandledRef = useRef<boolean>(false);
  
  // NEW: Ref to track if LLM is still streaming and pending TTS requests
  const isLlmStreamingRef = useRef<boolean>(false);
  const pendingTtsRequestsRef = useRef<number>(0);

  const resetStreamingRecognitionState = () => {
    speechFinalTranscriptRef.current = '';
    speechInterimTranscriptRef.current = '';
    recognitionStopRequestedRef.current = false;
    recognitionHasErrorRef.current = false;
    recognitionEndedRef.current = true;
    streamingTranscriptHandledRef.current = false;
  };

  const getStreamingTranscript = () =>
    [speechFinalTranscriptRef.current, speechInterimTranscriptRef.current]
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();

  const trySubmitStreamingTranscript = () => {
    const text = getStreamingTranscript();
    if (!text) return false;
    if (recognitionHasErrorRef.current && text.split(/\s+/).length < 2) return false;

    streamingTranscriptHandledRef.current = true;
    speechFinalTranscriptRef.current = '';
    speechInterimTranscriptRef.current = '';
    return submitUserPrompt(text, lastSpeechDurationRef.current, 'Streaming STT');
  };

  const startStreamingSpeechRecognition = () => {
    const RecognitionCtor = getSpeechRecognitionConstructor();
    if (!RecognitionCtor) {
      speechRecognitionModeRef.current = 'cloud';
      recognitionEndedRef.current = true;
      setTurnRealtimeFailureReason('browser_speech_recognition_unavailable');
      return false;
    }

    const recognition = new RecognitionCtor();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      let nextFinal = speechFinalTranscriptRef.current;
      let nextInterim = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i]?.[0]?.transcript?.trim();
        if (!transcript) continue;

        if (event.results[i].isFinal) {
          nextFinal = `${nextFinal} ${transcript}`.trim();
        } else {
          nextInterim = `${nextInterim} ${transcript}`.trim();
        }
      }

      speechFinalTranscriptRef.current = nextFinal;
      speechInterimTranscriptRef.current = nextInterim;

      const liveText = getStreamingTranscript();
      if (liveText) {
        setTranscript(liveText);
        markTranscriptVisible(liveText);
        hasSpokenRef.current = true;
        if (!speechStartRef.current) {
          speechStartRef.current = Date.now();
        }
        silenceStartRef.current = null;
      }
    };

    recognition.onerror = (event) => {
      recognitionHasErrorRef.current = true;
      console.warn('[Streaming STT Error]:', event?.error ?? event);
    };

    recognition.onend = () => {
      speechRecognitionRef.current = null;
      recognitionEndedRef.current = true;

      if (isListeningRef.current && !recognitionStopRequestedRef.current && !recognitionHasErrorRef.current) {
        window.setTimeout(() => {
          if (isListeningRef.current && !recognitionStopRequestedRef.current) {
            startStreamingSpeechRecognition();
          }
        }, 80);
      }
    };

    recognitionEndedRef.current = false;
    speechRecognitionRef.current = recognition;
    speechRecognitionModeRef.current = 'streaming';
    realtimeModeRef.current = 'browser';
    setTurnSttMode('browser');
    recognition.start();
    return true;
  };

  const startRealtimeTranscriptionSession = async (stream: MediaStream) => {
    if (typeof RTCPeerConnection === 'undefined') {
      setTurnRealtimeFailureReason('webrtc_not_supported');
      return false;
    }

    closeRealtimeTranscriptionSession();

    try {
      const peerConnection = new RTCPeerConnection();
      realtimePeerConnectionRef.current = peerConnection;
      realtimeModeRef.current = 'webrtc';

      stream.getAudioTracks().forEach((track) => {
        peerConnection.addTrack(track, stream);
      });

      const dataChannel = peerConnection.createDataChannel('oai-events');
      realtimeDataChannelRef.current = dataChannel;

      dataChannel.addEventListener('open', () => {
        setStatus('Listening realtime...');
      });

      dataChannel.addEventListener('message', (event) => {
        try {
          const payload = JSON.parse(event.data);

          if (payload?.type === 'conversation.item.input_audio_transcription.delta') {
            const itemId = payload.item_id || 'current';
            const previousText = realtimeTranscriptByItemIdRef.current.get(itemId) || '';
            const nextText = `${previousText}${payload.delta || ''}`.trim();
            realtimeTranscriptByItemIdRef.current.set(itemId, nextText);
            if (nextText) {
              setTranscript(nextText);
              markTranscriptVisible(nextText);
            }
          }

          if (payload?.type === 'conversation.item.input_audio_transcription.completed') {
            const completedText = (payload.transcript || '').trim();
            if (!completedText || handledListeningTurnRef.current) {
              return;
            }

            const itemId = payload.item_id || 'current';
            realtimeTranscriptByItemIdRef.current.set(itemId, completedText);
            handledListeningTurnRef.current = true;
            setTranscript(completedText);
            lastSpeechDurationRef.current = speechStartRef.current ? Math.max(0, Date.now() - speechStartRef.current) : lastSpeechDurationRef.current;
            markTranscriptVisible(completedText, lastSpeechDurationRef.current);
            const accepted = submitUserPrompt(completedText, lastSpeechDurationRef.current, 'Streaming STT');
            stopListening(false);

            if (!accepted && isSessionActiveRef.current) {
              checkAiFinishedAndResume();
            }
          }
        } catch (error) {
          console.warn('Realtime STT message parse error:', error);
        }
      });

      const handleRealtimeFailure = (reason: string) => {
        setTurnRealtimeFailureReason(reason);
        if (realtimeModeRef.current === 'webrtc') {
          realtimeModeRef.current = 'cloud';
        }
      };

      peerConnection.addEventListener('connectionstatechange', () => {
        if (['failed', 'disconnected', 'closed'].includes(peerConnection.connectionState)) {
          handleRealtimeFailure(`peer_${peerConnection.connectionState}`);
        }
      });

      dataChannel.addEventListener('close', () => handleRealtimeFailure('data_channel_closed'));
      dataChannel.addEventListener('error', () => handleRealtimeFailure('data_channel_error'));

      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      const response = await fetch('/api/realtime-stt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/sdp',
        },
        body: offer.sdp,
      });

      if (!response.ok) {
        throw new Error(`Realtime STT session failed: ${response.status} ${response.statusText}`);
      }

      const answerSdp = await response.text();
      await peerConnection.setRemoteDescription({ type: 'answer', sdp: answerSdp });
      setTurnSttMode('webrtc');
      return true;
    } catch (error) {
      console.warn('Realtime STT init failed, falling back:', error);
      setTurnRealtimeFailureReason(error instanceof Error ? error.message : String(error));
      closeRealtimeTranscriptionSession();
      return false;
    }
  };

  // Helper to check if AI is completely finished
  const checkAiFinishedAndResume = () => {
    if (!isLlmStreamingRef.current && pendingTtsRequestsRef.current === 0 && audioQueueRef.current.length === 0 && !isPlayingRef.current) {
      finalizeTurnTelemetry('completed');
      stopPassiveBargeInMonitor();
      setIsBargeInIntent(false);
      setIsPreSpeechCue(false);
      setIsAiSpeaking(false);
      setIsThinking(false);
      setStatus('Ready');
      
      if (resumeTimeoutRef.current) window.clearTimeout(resumeTimeoutRef.current);
      
      // Let the learner fully hear the ending before re-opening the mic.
      if (isSessionActiveRef.current && !isListeningRef.current) {
        resumeTimeoutRef.current = window.setTimeout(() => {
          if (
            isSessionActiveRef.current &&
            !isListeningRef.current &&
            !isPlayingRef.current &&
            !isAiSpeakingRef.current &&
            !isLlmStreamingRef.current &&
            pendingTtsRequestsRef.current === 0 &&
            audioQueueRef.current.length === 0
          ) {
            void startListening({ interruptCurrentAi: false });
          }
        }, AI_TO_USER_RESUME_MS);
      }
    }
  };

  const clearPreSpeechCue = () => {
    if (preSpeechCueTimeoutRef.current) {
      window.clearTimeout(preSpeechCueTimeoutRef.current);
      preSpeechCueTimeoutRef.current = null;
    }
    setIsPreSpeechCue(false);
  };

  const clearBargeInIntent = () => {
    if (bargeInIntentTimeoutRef.current) {
      window.clearTimeout(bargeInIntentTimeoutRef.current);
      bargeInIntentTimeoutRef.current = null;
    }
    setIsBargeInIntent(false);
  };

  const stopPassiveBargeInMonitor = () => {
    if (passiveBargeInFrameRef.current) {
      cancelAnimationFrame(passiveBargeInFrameRef.current);
      passiveBargeInFrameRef.current = null;
    }

    passiveBargeInSpeechStartRef.current = null;
    passiveBargeInSilenceStartRef.current = null;
    passiveBargeInAnalyserRef.current = null;

    if (passiveBargeInContextRef.current) {
      passiveBargeInContextRef.current.close();
      passiveBargeInContextRef.current = null;
    }

    if (passiveBargeInStreamRef.current) {
      passiveBargeInStreamRef.current.getTracks().forEach(track => track.stop());
      passiveBargeInStreamRef.current = null;
    }

    passiveBargeInStartingRef.current = false;
  };

  const startPassiveBargeInMonitor = async () => {
    if (
      passiveBargeInStartingRef.current ||
      passiveBargeInStreamRef.current ||
      softBargeInInProgressRef.current ||
      isListeningRef.current ||
      !isSessionActiveRef.current ||
      (!isAiSpeakingRef.current && !isThinkingRef.current) ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      return;
    }

    passiveBargeInStartingRef.current = true;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });

      if (
        softBargeInInProgressRef.current ||
        isListeningRef.current ||
        !isSessionActiveRef.current ||
        (!isAiSpeakingRef.current && !isThinkingRef.current)
      ) {
        stream.getTracks().forEach(track => track.stop());
        passiveBargeInStartingRef.current = false;
        return;
      }

      passiveBargeInStreamRef.current = stream;

      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      passiveBargeInContextRef.current = new AudioContextClass();
      const source = passiveBargeInContextRef.current.createMediaStreamSource(stream);
      passiveBargeInAnalyserRef.current = passiveBargeInContextRef.current.createAnalyser();
      passiveBargeInAnalyserRef.current.fftSize = 256;
      source.connect(passiveBargeInAnalyserRef.current);

      const detectUserBargeIn = () => {
        const analyser = passiveBargeInAnalyserRef.current;
        if (!analyser) return;

        if (
          softBargeInInProgressRef.current ||
          isListeningRef.current ||
          !isSessionActiveRef.current ||
          (!isAiSpeakingRef.current && !isThinkingRef.current)
        ) {
          stopPassiveBargeInMonitor();
          return;
        }

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);
        const sum = dataArray.reduce((a, b) => a + b, 0);
        const avg = sum / dataArray.length;
        const now = Date.now();

        if (avg > VOICE_BARGE_IN_VOLUME_THRESHOLD) {
          passiveBargeInSilenceStartRef.current = null;
          if (!passiveBargeInSpeechStartRef.current) {
            passiveBargeInSpeechStartRef.current = now;
          } else if (
            now - passiveBargeInSpeechStartRef.current >= VOICE_BARGE_IN_TRIGGER_MS &&
            now - passiveBargeInLastTriggerRef.current >= VOICE_BARGE_IN_COOLDOWN_MS
          ) {
            passiveBargeInLastTriggerRef.current = now;
            console.info('[Voice Barge-In] Sustained user speech detected during AI response.');
            void runSoftBargeIn('voice');
            return;
          }
        } else if (passiveBargeInSpeechStartRef.current) {
          if (!passiveBargeInSilenceStartRef.current) {
            passiveBargeInSilenceStartRef.current = now;
          } else if (now - passiveBargeInSilenceStartRef.current >= VOICE_BARGE_IN_RELEASE_MS) {
            passiveBargeInSpeechStartRef.current = null;
            passiveBargeInSilenceStartRef.current = null;
          }
        }

        passiveBargeInFrameRef.current = requestAnimationFrame(detectUserBargeIn);
      };

      passiveBargeInFrameRef.current = requestAnimationFrame(detectUserBargeIn);
    } catch (error) {
      console.warn('[Voice Barge-In] Passive monitor unavailable:', error);
    } finally {
      passiveBargeInStartingRef.current = false;
    }
  };

  const runPreSpeechCue = async (responseId: number) => {
    if (responseId !== activeResponseIdRef.current || hasStartedSpeechForResponseRef.current) {
      return true;
    }

    clearPreSpeechCue();
    setIsPreSpeechCue(true);

    await new Promise<void>((resolve) => {
      preSpeechCueTimeoutRef.current = window.setTimeout(() => {
        preSpeechCueTimeoutRef.current = null;
        resolve();
      }, PRE_SPEECH_CUE_MS);
    });

    if (responseId !== activeResponseIdRef.current) {
      setIsPreSpeechCue(false);
      return false;
    }

    setIsPreSpeechCue(false);
    return true;
  };

  // Interrupt AI playback and generation
  const interruptAi = () => {
    finalizeTurnTelemetry('aborted', { error: 'Interrupted by user or new turn.' });
    stopPassiveBargeInMonitor();
    clearBargeInIntent();
    clearPreSpeechCue();
    activeResponseIdRef.current += 1;
    livePreviewResponseIdRef.current = activeResponseIdRef.current;
    hasStartedSpeechForResponseRef.current = false;
    currentSentenceIndexRef.current = 0;
    nextSentenceToPlayRef.current = 0;
    pendingSentenceAudioRef.current.clear();
    isLlmStreamingRef.current = false;
    pendingTtsRequestsRef.current = 0;
    if (typewriterIntervalRef.current) {
      clearInterval(typewriterIntervalRef.current);
      typewriterIntervalRef.current = null;
    }
    if (resumeTimeoutRef.current) {
      window.clearTimeout(resumeTimeoutRef.current);
      resumeTimeoutRef.current = null;
    }
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
    setIsThinking(false);
    setLiveAiPreview('');
    closeRealtimeTranscriptionSession();
    if (speechRecognitionRef.current) {
      recognitionStopRequestedRef.current = true;
      try {
        speechRecognitionRef.current.abort();
      } catch {
        // noop
      }
      speechRecognitionRef.current = null;
    }
    resetStreamingRecognitionState();
    llmWorkerRef.current?.postMessage({ type: 'ABORT_GENERATION' });
  };

  const runSoftBargeIn = async (source: 'manual' | 'voice' = 'manual') => {
    if (isListeningRef.current || softBargeInInProgressRef.current) return;

    if (source === 'manual' && isBargeInIntent) {
      clearBargeInIntent();
      setStatus(isAiSpeakingRef.current || isThinkingRef.current ? 'Continuing response...' : 'Ready');
      return;
    }

    softBargeInInProgressRef.current = true;
    stopPassiveBargeInMonitor();
    triggerHaptic(source === 'voice' ? 'medium' : 'light');
    setIsBargeInIntent(true);
    setStatus(source === 'voice' ? 'I hear you. Taking the turn...' : 'Taking the turn...');

    try {
      await new Promise<void>((resolve) => {
        bargeInIntentTimeoutRef.current = window.setTimeout(() => {
          bargeInIntentTimeoutRef.current = null;
          resolve();
        }, SOFT_BARGE_IN_CUE_MS);
      });

      if (!isAiSpeakingRef.current && !isThinkingRef.current && !isLlmStreamingRef.current && !isPlayingRef.current) {
        clearBargeInIntent();
        return;
      }

      clearBargeInIntent();
      isSessionActiveRef.current = true;
      interruptAi();
      await startListening({ interruptCurrentAi: false });
    } finally {
      softBargeInInProgressRef.current = false;
    }
  };

  // Helper to play TTS audio sequentially
  const playNextAudio = async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      if (audioQueueRef.current.length === 0 && !isPlayingRef.current) {
        checkAiFinishedAndResume();
      }
      return;
    }

    isPlayingRef.current = true;
    setIsAiSpeaking(true);
    setIsThinking(false); // 🛑 Stop thinking animation when speaking starts

    const audioData = audioQueueRef.current.shift();
    if (!audioData) {
      isPlayingRef.current = false;
      setIsAiSpeaking(false);
      return;
    }

    const shouldRunPreSpeechCue =
      audioData.responseId === activeResponseIdRef.current &&
      !hasStartedSpeechForResponseRef.current;

    if (shouldRunPreSpeechCue) {
      const canContinue = await runPreSpeechCue(audioData.responseId);
      if (!canContinue) {
        isPlayingRef.current = false;
        currentAudioSourceRef.current = null;
        if (audioQueueRef.current.length === 0) {
          checkAiFinishedAndResume();
        } else {
          playNextAudio();
        }
        return;
      }
    }

    if (audioData.responseId === activeResponseIdRef.current) {
      hasStartedSpeechForResponseRef.current = true;
      setLiveAiPreview('');
      markAssistantOutput(audioData.textOnly);
    }

    if (!playbackContextRef.current) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      playbackContextRef.current = new AudioContextClass(); // Use default sample rate for decoding
    }

    const audioCtx = playbackContextRef.current;
    if (audioCtx.state === 'suspended') audioCtx.resume();

    try {
      if (audioData.textOnly || !audioData.buffer) {
        const words = audioData.text.split(' ');
        const timePerWord = 85;
        let wordIndex = 0;

        if (typewriterIntervalRef.current) clearInterval(typewriterIntervalRef.current);

        typewriterIntervalRef.current = window.setInterval(() => {
          if (audioData.responseId !== activeResponseIdRef.current) {
            if (typewriterIntervalRef.current) {
              clearInterval(typewriterIntervalRef.current);
              typewriterIntervalRef.current = null;
            }
            return;
          }
          if (wordIndex < words.length) {
            const word = words[wordIndex];
            setAiResponse(prev => {
              const trimmed = prev.trim();
              return trimmed ? trimmed + ' ' + word : word;
            });
            wordIndex++;
          } else if (typewriterIntervalRef.current) {
            clearInterval(typewriterIntervalRef.current);
            typewriterIntervalRef.current = null;
            isPlayingRef.current = false;
            currentAudioSourceRef.current = null;
            if (audioQueueRef.current.length === 0) {
              checkAiFinishedAndResume();
            } else {
              playNextAudio();
            }
          }
        }, timePerWord);
        return;
      }

      // Decode the MP3 array buffer from OpenAI
      // Cast it back to ArrayBuffer before decoding
      const audioBuffer = await audioCtx.decodeAudioData(audioData.buffer);
      if (audioData.responseId !== activeResponseIdRef.current) {
        isPlayingRef.current = false;
        playNextAudio();
        return;
      }
      
      // 🛑 Teleprompter effect: Type out the sentence while audio plays
      const words = audioData.text.split(' ');
      // Estimate time per word based on audio duration (fallback to 100ms if very short)
      const timePerWord = Math.max((audioBuffer.duration * 1000) / (words.length || 1), 50);
      let wordIndex = 0;
      
      if (typewriterIntervalRef.current) clearInterval(typewriterIntervalRef.current);
      
      typewriterIntervalRef.current = window.setInterval(() => {
        if (audioData.responseId !== activeResponseIdRef.current) {
          if (typewriterIntervalRef.current) {
            clearInterval(typewriterIntervalRef.current);
            typewriterIntervalRef.current = null;
          }
          return;
        }
        if (wordIndex < words.length) {
          const word = words[wordIndex];
          setAiResponse(prev => {
            const trimmed = prev.trim();
            return trimmed ? trimmed + ' ' + word : word;
          });
          wordIndex++;
        } else {
          if (typewriterIntervalRef.current) {
            clearInterval(typewriterIntervalRef.current);
            typewriterIntervalRef.current = null;
          }
        }
      }, timePerWord);

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);
      currentAudioSourceRef.current = source;
      
      source.onended = () => {
        if (audioData.responseId !== activeResponseIdRef.current) {
          currentAudioSourceRef.current = null;
          isPlayingRef.current = false;
          return;
        }

        // Ensure all words are fully displayed if audio finishes before typewriter
        if (typewriterIntervalRef.current) {
          clearInterval(typewriterIntervalRef.current);
          typewriterIntervalRef.current = null;
        }
        
        // Append any remaining words that the typewriter missed
        if (wordIndex < words.length) {
          const remainingWords = words.slice(wordIndex).join(' ');
          setAiResponse(prev => {
            const trimmed = prev.trim();
            return trimmed ? trimmed + ' ' + remainingWords : remainingWords;
          });
        }
        
        currentAudioSourceRef.current = null;
        isPlayingRef.current = false;
        if (audioQueueRef.current.length === 0) {
          checkAiFinishedAndResume();
        } else {
          playNextAudio();
        }
      };

      source.start(0);
    } catch (error) {
      console.error('Audio playback error:', error);
      if (audioData.responseId === activeResponseIdRef.current && !audioData.textOnly) {
        audioQueueRef.current.unshift({
          ...audioData,
          buffer: null,
          textOnly: true,
        });
      }
      isPlayingRef.current = false;
      playNextAudio();
    }
  };

  useEffect(() => {
    // Initialize Web Worker for LLM
    llmWorkerRef.current = new Worker(new URL('./workers/llm.worker.ts', import.meta.url), { type: 'module' });

    // Handle messages from LLM Worker
    llmWorkerRef.current.onmessage = (e: MessageEvent) => {
      const { type, payload } = e.data;
      
      switch (type) {
        case 'LLM_READY':
          console.log('[LLM Worker Ready]');
          setStatus('Ready (Zero Lag UI) - Connected to Cloud AI');
          setIsReady(true);
          break;
          
        case 'TEXT_CHUNK':
          markLlmChunk(payload?.text ?? '');
          if (
            payload?.responseId === activeResponseIdRef.current &&
            livePreviewResponseIdRef.current === payload.responseId &&
            !hasStartedSpeechForResponseRef.current
          ) {
            setLiveAiPreview(prev => prev + (payload.text ?? ''));
          }
          break;

        case 'SESSION_METRICS':
          setSessionMetrics(payload);
          setTutorMode(payload.tutorMode);
          break;
          
        case 'SENTENCE_READY':
          console.log('[LLM Sentence Ready for TTS]:', payload);
          if (payload.responseId !== activeResponseIdRef.current) {
            break;
          }
          synthesizeWithOpenAI(payload.text, payload.responseId, currentSentenceIndexRef.current);
          currentSentenceIndexRef.current += 1;
          break;
          
        case 'STREAM_END':
          console.log('[LLM Stream End]');
          if (payload?.responseId !== activeResponseIdRef.current) {
            break;
          }
          isLlmStreamingRef.current = false;
          checkAiFinishedAndResume();
          break;
          
        case 'ERROR':
          console.error('[LLM Worker Error]:', payload);
          if (!payload?.responseId || payload.responseId === activeResponseIdRef.current) {
            const nextStatus = payload?.retryable
              ? 'Gemini is temporarily busy. You can try again in a moment.'
              : `LLM Error: ${payload.message ?? payload}`;
            setStatus(nextStatus);
            setIsThinking(false);
            setLiveAiPreview('');
            finalizeTurnTelemetry(payload?.retryable ? 'retryable_error' : 'error', {
              error: payload?.message ?? String(payload),
            });
            if (payload?.retryable && isSessionActiveRef.current) {
              checkAiFinishedAndResume();
            }
          }
          break;
          
        default:
          console.log('Unknown message from LLM Worker:', e.data);
      }
    };

    // Send the starting signal to check LLM API
    llmWorkerRef.current.postMessage({ type: 'INIT_LLM' });

    // Cleanup workers on unmount
    return () => {
      stopPassiveBargeInMonitor();
      clearBargeInIntent();
      clearPreSpeechCue();
      if (vadFrameRef.current) cancelAnimationFrame(vadFrameRef.current);
      if (vadContextRef.current) vadContextRef.current.close();
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      closeRealtimeTranscriptionSession();
      if (speechRecognitionRef.current) {
        try {
          speechRecognitionRef.current.abort();
        } catch {
          // noop
        }
      }
      llmWorkerRef.current?.terminate();
    };
  }, []);

  useEffect(() => {
    isListeningRef.current = isListening;
  }, [isListening]);

  useEffect(() => {
    isAiSpeakingRef.current = isAiSpeaking;
  }, [isAiSpeaking]);

  useEffect(() => {
    isThinkingRef.current = isThinking;
  }, [isThinking]);

  useEffect(() => {
    if (
      !isListening &&
      !isBargeInIntent &&
      isSessionActiveRef.current &&
      (isAiSpeaking || isThinking)
    ) {
      void startPassiveBargeInMonitor();
    } else {
      stopPassiveBargeInMonitor();
    }
  }, [isListening, isBargeInIntent, isAiSpeaking, isThinking]);

  const getDefaultLessonFocus = (mode: TutorMode) => LESSON_FOCUS_OPTIONS[mode][0];

  const handleTutorModeChange = (mode: TutorMode) => {
    const nextFocus = getDefaultLessonFocus(mode);
    setTutorMode(mode);
    llmWorkerRef.current?.postMessage({ type: 'SET_TUTOR_MODE', payload: { mode, lessonFocus: nextFocus } });
  };

  const handleBudgetReset = () => {
    llmWorkerRef.current?.postMessage({ type: 'RESET_SESSION_BUDGET' });
    setSessionMetrics(prev => ({
      ...prev,
      estimatedInputTokens: 0,
      estimatedOutputTokens: 0,
      estimatedTotalTokens: 0,
      memoryItems: 0,
    }));
  };

  const handleCompressMemory = () => {
    llmWorkerRef.current?.postMessage({ type: 'COMPRESS_HISTORY' });
  };

  const handleMicClick = async () => {
    triggerHaptic('medium');
    
    if (isListening) {
      // Manual Stop & Send
      console.log('[UI] Manual stop & process');
      isSessionActiveRef.current = false; // Stop the continuous loop
      stopListening(true);
    } else if (isAiSpeaking || isThinking) {
      console.log('[UI] Soft barge-in intent');
      await runSoftBargeIn();
    } else {
      // Start Listening
      isSessionActiveRef.current = true;
      await startListening({ interruptCurrentAi: false });
    }
  };

  const startListening = async (options?: { interruptCurrentAi?: boolean }) => {
    const shouldInterruptCurrentAi = options?.interruptCurrentAi ?? true;
    if (shouldInterruptCurrentAi) {
      interruptAi();
    } else if (resumeTimeoutRef.current) {
      window.clearTimeout(resumeTimeoutRef.current);
      resumeTimeoutRef.current = null;
    }

    try {
      stopPassiveBargeInMonitor();
      clearBargeInIntent();
      if (turnTelemetryRef.current && !turnTelemetryRef.current.status) {
        finalizeTurnTelemetry('aborted', { error: 'Superseded by a new listening turn.' });
      }
      beginTurnTelemetry();
      setStatus('Listening...');
      setIsListening(true);
      hasSpokenRef.current = false;
      silenceStartRef.current = null;
      speechStartRef.current = null;
      lastSpeechDurationRef.current = 0;
      shouldProcessAudioRef.current = true;
      handledListeningTurnRef.current = false;
      realtimeTranscriptByItemIdRef.current.clear();
      realtimeModeRef.current = 'cloud';
      setTranscript('');
      resetStreamingRecognitionState();
      speechRecognitionModeRef.current = 'cloud';
      
      // 🛑 ПРИЧИНА 3 ИСПРАВЛЕНА: Разблокировка AudioContext на iPhone (Safari)
      if (!playbackContextRef.current) {
        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        playbackContextRef.current = new AudioContextClass();
        
        // Play 1ms of silence to unlock the audio context
        const silentBuffer = playbackContextRef.current.createBuffer(1, 1, 22050);
        const source = playbackContextRef.current.createBufferSource();
        source.buffer = silentBuffer;
        source.connect(playbackContextRef.current.destination);
        source.start(0);
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }, 
        video: false 
      });
      mediaStreamRef.current = stream;

      const hasRealtimeSession = await startRealtimeTranscriptionSession(stream);

      if (!hasRealtimeSession && startStreamingSpeechRecognition()) {
        setStatus('Listening live...');
      }
      
      const preferredMimeType = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
      ].find((mimeType) => typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported?.(mimeType));

      const mediaRecorder = preferredMimeType
        ? new MediaRecorder(stream, {
            mimeType: preferredMimeType,
            audioBitsPerSecond: 24000,
          })
        : new MediaRecorder(stream);
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
          if (!handledListeningTurnRef.current) {
            setIsThinking(false);
          }
          handledListeningTurnRef.current = false;
          return; // Discard audio if stopped forcefully
        }
        
        try {
          if (speechRecognitionModeRef.current === 'streaming') {
            await new Promise(resolve => window.setTimeout(resolve, recognitionEndedRef.current ? 40 : 260));
            if (streamingTranscriptHandledRef.current || trySubmitStreamingTranscript()) {
              return;
            }
          }

          setIsThinking(true); // Start thinking animation
          setStatus('Sending to OpenAI STT...');
          const blob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType });
          // FAST CLOUD STT: Send blob directly to OpenAI
          await transcribeWithOpenAI(blob);
        } catch (error) {
          console.error('Audio processing error:', error);
          setStatus('Error: Failed to process audio');
          setIsThinking(false);
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
        // If we manually stopped listening, abort VAD loop
        if (!analyserRef.current) return;
        
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        
        const sum = dataArray.reduce((a, b) => a + b, 0);
        const avg = sum / dataArray.length;

        // Update volume for visualizer (mapping 0-255 to a 1.0 - 1.5 scale)
        const normalizedVolume = 1 + (avg / 255) * 0.8;
        audioVolumeRef.current = normalizedVolume;

        // Force React to re-render the sphere scale only (using a CSS variable directly on the DOM node for performance)
        const sphereEl = document.querySelector('.orb') as HTMLElement | null;
        if (sphereEl) {
          sphereEl.style.transform = `scale(${normalizedVolume})`;
        }

        // VAD Logic
        // 🛑 COMPLETELY IGNORE MIC IF AI IS SPEAKING OR THINKING
        if (isAiSpeaking || isThinking || isPlayingRef.current || isLlmStreamingRef.current) {
          hasSpokenRef.current = false;
          speechStartRef.current = null;
          silenceStartRef.current = null;
        } else {
          if (avg > 18) { 
            // Volume threshold exceeded (User is speaking)
            if (!hasSpokenRef.current) {
              hasSpokenRef.current = true;
              speechStartRef.current = Date.now();
            }
            silenceStartRef.current = null;
          } else if (hasSpokenRef.current) {
            // User has spoken, now detecting silence
            if (!silenceStartRef.current) {
              silenceStartRef.current = Date.now();
            } else if (Date.now() - silenceStartRef.current > USER_END_OF_TURN_MS) { 
              // A calm pause means the learner has likely finished this turn.
              // Now check if the user actually spoke a full sentence, or just sneezed/coughed.
              const speechDuration = silenceStartRef.current - (speechStartRef.current || 0);

              if (realtimeModeRef.current === 'webrtc' && Date.now() - silenceStartRef.current < USER_END_OF_TURN_MS + 2200) {
                lastSpeechDurationRef.current = speechDuration;
                vadFrameRef.current = requestAnimationFrame(checkSilence);
                return;
              }
              
              if (speechDuration < 800) {
                // Short noise, sneeze, cough. Ignore it.
                console.log(`[VAD] Ignored short noise (${speechDuration}ms)`);
                hasSpokenRef.current = false;
                speechStartRef.current = null;
                silenceStartRef.current = null;
              } else {
                console.log(`[VAD] Valid speech detected (${speechDuration}ms). Stopping mic to process...`);
                lastSpeechDurationRef.current = speechDuration;
                triggerHaptic('heavy'); 
                stopListening(true);
                return;
              }
            }
          }
        }
        vadFrameRef.current = requestAnimationFrame(checkSilence);
      };

      mediaRecorder.start(250);
      checkSilence();
      
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setStatus('Error: Could not access microphone');
    }
  };

  const stopListening = (process: boolean) => {
    stopPassiveBargeInMonitor();
    shouldProcessAudioRef.current = process;
    if (process) {
      const now = Date.now();
      const speechStart = speechStartRef.current;
      lastSpeechDurationRef.current = speechStart ? Math.max(0, now - speechStart) : lastSpeechDurationRef.current;
    }

    if (speechRecognitionRef.current) {
      recognitionStopRequestedRef.current = true;
      try {
        speechRecognitionRef.current.stop();
      } catch {
        // noop
      }
    }
    
    if (vadFrameRef.current) {
      cancelAnimationFrame(vadFrameRef.current);
      vadFrameRef.current = null;
    }
    if (vadContextRef.current) {
      vadContextRef.current.close();
      vadContextRef.current = null;
    }

    closeRealtimeTranscriptionSession();

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    setIsListening(false);
    
    // Reset visualizer scale immediately
    const sphereEl = document.querySelector('.orb') as HTMLElement | null;
    if (sphereEl) {
      sphereEl.style.transform = `scale(1)`;
    }
  };

  // Generate random particles for background
  const particles = useMemo(() => {
    return Array.from({ length: 42 }).map((_, i) => ({
      id: i,
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      delay: `${Math.random() * 6}s`,
      duration: `${9 + Math.random() * 16}s`,
      size: `${1.8 + Math.random() * 4.6}px`,
      opacity: `${0.28 + Math.random() * 0.48}`,
      glow: `${8 + Math.random() * 18}px`,
      blur: `${Math.random() * 1.5}px`,
      drift: `${-16 + Math.random() * 32}px`,
      tint: Math.random() > 0.62 ? 'rgba(216, 180, 254, 0.95)' : Math.random() > 0.35 ? 'rgba(125, 211, 252, 0.9)' : 'rgba(255, 255, 255, 0.95)'
    }));
  }, []);

  const isBusy = isListening || isAiSpeaking || isThinking || isBargeInIntent;
  const budgetProgress = Math.min(100, Math.round((sessionMetrics.estimatedTotalTokens / sessionMetrics.tokenBudget) * 100));
  const showBudgetWarning = budgetProgress >= 75;
  const isBudgetCritical = budgetProgress >= 90;
  const hasTranscript = Boolean(transcript);
  const hasAiResponse = Boolean(aiResponse);
  const hasLiveAiPreview = Boolean(liveAiPreview && !hasAiResponse);
  const visibleAiText = hasAiResponse ? aiResponse : liveAiPreview;
  const hasVisibleAiText = Boolean(visibleAiText);
  const subtitleLayoutClass = hasTranscript && hasVisibleAiText ? 'dual-card' : hasTranscript || hasVisibleAiText ? 'single-card' : 'idle';
  const sessionHealthLabel = isBudgetCritical ? 'High Budget' : showBudgetWarning ? 'Budget Warning' : '';

  return (
    <div className={`app-container ${isListening ? 'state-listening' : ''} ${isAiSpeaking ? 'state-speaking' : ''} ${isThinking ? 'state-thinking' : ''} ${isPreSpeechCue ? 'state-pre-speech' : ''} ${isBargeInIntent ? 'state-barge-in' : ''}`}>
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
              height: p.size,
              opacity: p.opacity,
              '--particle-glow': p.glow,
              '--particle-blur': p.blur,
              '--particle-drift': p.drift,
              '--particle-color': p.tint,
            } as CSSProperties}
          />
        ))}
      </div>
      <div className="scene-ambient scene-ambient-primary" />
      <div className="scene-ambient scene-ambient-secondary" />

      <div className="hero-shell scene-layer scene-layer-top">
        <div className="control-deck">
          <div className="top-tags tutor-mode-row">
            {TUTOR_MODE_OPTIONS.map(option => (
              <button
                key={option.value}
                className={`tag tutor-tag ${tutorMode === option.value ? 'active' : ''}`}
                onClick={() => handleTutorModeChange(option.value)}
                type="button"
                disabled={isThinking}
                title={option.hint}
              >
                <span>{option.label}</span>
              </button>
            ))}
          </div>

          {/* Greeting and Status Header */}
          <div className="status-header">
            <div className="greeting-name">Zero AI by Nursultan and Aliya</div>
            <div className="main-prompt">
              {isListening ? "I'M LISTENING" : isBargeInIntent ? "JUMPING IN" : isAiSpeaking ? "ZERO AI" : isThinking ? "THINKING..." : "SAY SOMETHING"}
            </div>
            <div className="status-text">{status}</div>
          </div>

          <div className="session-panel">
            <div className="session-panel-header">
              <div className="session-panel-title-group">
                <div className="session-panel-title">Adaptive Tutor Memory</div>
              </div>
              {showBudgetWarning ? (
                <div className={`session-health-badge ${isBudgetCritical ? 'critical' : 'warning'}`}>
                  {sessionHealthLabel}
                </div>
              ) : null}
            </div>
            <div className="budget-bar">
              <div className="budget-fill" style={{ width: `${budgetProgress}%` }} />
            </div>
            {showBudgetWarning ? (
              <div className={`budget-warning ${isBudgetCritical ? 'critical' : ''}`}>
                <div className="budget-warning-text">
                  {isBudgetCritical
                    ? 'Session budget is very high. Reset now or compress memory to save tokens.'
                    : 'Session budget reached 75%. You can compress memory or reset the session to save tokens.'}
                </div>
                <div className="budget-warning-actions">
                  <button className="secondary-action-button" type="button" onClick={handleCompressMemory}>
                    Compress Memory
                  </button>
                  <button className="reset-budget-button" type="button" onClick={handleBudgetReset}>
                    Reset Session
                  </button>
                </div>
              </div>
            ) : (
              <button className="reset-budget-button" type="button" onClick={handleBudgetReset}>
                Reset Session
              </button>
            )}
          </div>
        </div>
      </div>
      
      <div className={`orb-container scene-layer scene-layer-orb ${isListening ? 'active' : ''} ${isAiSpeaking ? 'ai-speaking' : ''} ${isThinking ? 'thinking' : ''} ${isPreSpeechCue ? 'pre-speech' : ''} ${isBargeInIntent ? 'barge-in' : ''}`}>
        <div className="orb">
          <div className="petal petal-1"></div>
          <div className="petal petal-2"></div>
          <div className="petal petal-3"></div>
          <div className="petal petal-4"></div>
          <div className="orb-inner"></div>
          <div className="orb-core"></div>
        </div>
      </div>

      <div className={`text-container subtitle-stage scene-layer scene-layer-subtitles ${hasTranscript || hasVisibleAiText ? 'engaged' : ''} ${subtitleLayoutClass} ${isPreSpeechCue ? 'pre-speech' : ''} ${isBargeInIntent ? 'barge-in' : ''}`}>
        {hasTranscript && (
          <div className="user-text transcript-card">
            <div className="text-label">You said</div>
            {transcript}
          </div>
        )}
        <div className={`ai-text response-card ${hasVisibleAiText ? 'has-content' : 'is-placeholder'} ${hasLiveAiPreview ? 'is-live-preview' : ''}`}>
          <div className="text-label ai-label">Zero AI</div>
          {visibleAiText ? visibleAiText : <span className="placeholder-text">Wait for response...</span>}
        </div>
      </div>

      <div className="bottom-controls mic-zone scene-layer scene-layer-mic">
        <div className="mic-wrapper">
          {isListening && (
            <>
              <div className="ripple ripple-1"></div>
              <div className="ripple ripple-2"></div>
              <div className="ripple ripple-3"></div>
            </>
          )}
          <button 
            className={`mic-button ${isBusy ? 'recording' : ''} ${isBargeInIntent ? 'barge-in' : ''}`}
            onClick={handleMicClick}
            disabled={!isReady}
          >
            {isBusy ? (
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
