// llm.worker.ts - API & Logic Layer
// This worker handles interactions with Gemini API, Supabase, and heavy business logic.

// ==========================================
// 1. State Management
// ==========================================
interface ChatMessage {
  role: string;
  parts: Array<{ text: string }>;
}

type TutorMode = 'grammar' | 'free_speaking';

interface SessionMetrics {
  tutorMode: TutorMode;
  lessonFocus: string;
  estimatedInputTokens: number;
  estimatedOutputTokens: number;
  estimatedTotalTokens: number;
  tokenBudget: number;
  memoryItems: number;
}

const DEFAULT_FOCUS: Record<TutorMode, string> = {
  grammar: 'general grammar',
  free_speaking: 'general conversation',
};

const state = {
  messages: [] as ChatMessage[],
  isAborted: false,
  tutorMode: 'free_speaking' as TutorMode,
  lessonFocus: DEFAULT_FOCUS.free_speaking,
  lessonMemory: [] as string[],
  estimatedInputTokens: 0,
  estimatedOutputTokens: 0,
};

// ==========================================
// NEW: Sliding Window Context Management
// Prevents context overflow and memory leaks
// ==========================================
const SESSION_TOKEN_BUDGET = 1800;
const MAX_RAW_MESSAGES = 4;
const MAX_LESSON_MEMORY_ITEMS = 5;
const TARGET_CONTEXT_TOKENS = 320;
const SOFT_SPLIT_TARGET_CHARS = 110;
const SOFT_SPLIT_MIN_CHARS = 68;

function isNearBudget() {
  return state.estimatedInputTokens + state.estimatedOutputTokens >= SESSION_TOKEN_BUDGET * 0.75;
}

function estimateTokens(text: string) {
  return Math.ceil(text.trim().length / 4);
}

function getMessageText(message: ChatMessage) {
  return message.parts.map(part => part.text).join(' ').trim();
}

function messageImportance(message: ChatMessage) {
  const text = getMessageText(message);
  let score = message.role === 'model' ? 2 : 1;

  if (/(mistake|correct|correction|grammar|pronunciation|tense|article|preposition|stress|sound|repeat|remember|goal|struggle|improve)/i.test(text)) {
    score += 3;
  }
  if (/\?/.test(text)) {
    score += 1;
  }
  if (text.length > 120) {
    score += 1;
  }

  return score;
}

function compressText(text: string) {
  return text.replace(/\s+/g, ' ').trim().slice(0, 72);
}

function buildMemoryNote(message: ChatMessage) {
  const text = compressText(getMessageText(message));
  if (!text) return '';
  return message.role === 'user' ? `Learner: ${text}` : `Tutor: ${text}`;
}

function compressHistoryIfNeeded(force = false) {
  const nearBudget = isNearBudget();
  const isFreeSpeaking = state.tutorMode === 'free_speaking';
  const maxRawMessages = force || nearBudget ? (isFreeSpeaking ? 1 : 2) : (isFreeSpeaking ? 2 : MAX_RAW_MESSAGES);
  const maxLessonMemoryItems = force || nearBudget ? (isFreeSpeaking ? 2 : 3) : (isFreeSpeaking ? 3 : MAX_LESSON_MEMORY_ITEMS);
  const targetContextTokens = force || nearBudget ? (isFreeSpeaking ? 160 : 220) : (isFreeSpeaking ? 220 : TARGET_CONTEXT_TOKENS);
  const historyTokens =
    estimateTokens(state.lessonMemory.join(' ')) +
    state.messages.reduce((total, message) => total + estimateTokens(getMessageText(message)), 0);

  if (state.messages.length <= maxRawMessages && historyTokens <= targetContextTokens) {
    return;
  }

  const rawMessagesToKeep = state.messages.slice(-maxRawMessages);
  const messagesToCompress = state.messages.slice(0, Math.max(0, state.messages.length - maxRawMessages));

  const rankedNotes = messagesToCompress
    .map((message) => ({ note: buildMemoryNote(message), score: messageImportance(message) }))
    .filter(item => item.note)
    .sort((a, b) => b.score - a.score)
    .slice(0, isFreeSpeaking ? 1 : 2)
    .map(item => item.note);

  if (rankedNotes.length > 0) {
    const nextMemory = [...state.lessonMemory, ...rankedNotes];
    const dedupedMemory = Array.from(new Set(nextMemory));
    state.lessonMemory = dedupedMemory.slice(-maxLessonMemoryItems);
  }

  state.messages = rawMessagesToKeep;
}

function rollbackPendingUserTurn() {
  const lastMessage = state.messages[state.messages.length - 1];
  if (lastMessage?.role === 'user') {
    state.messages.pop();
  }
}

function buildSessionMetrics(): SessionMetrics {
  const estimatedTotalTokens = state.estimatedInputTokens + state.estimatedOutputTokens;

  return {
    tutorMode: state.tutorMode,
    lessonFocus: state.lessonFocus,
    estimatedInputTokens: state.estimatedInputTokens,
    estimatedOutputTokens: state.estimatedOutputTokens,
    estimatedTotalTokens,
    tokenBudget: SESSION_TOKEN_BUDGET,
    memoryItems: state.lessonMemory.length,
  };
}

function publishSessionMetrics() {
  self.postMessage({ type: 'SESSION_METRICS', payload: buildSessionMetrics() });
}


// ==========================================
// 2. Sentence Splitter (Buffering)
// ==========================================
// Matches sentence endings: ., ?, !, followed by a space or end of string
// 🛑 ПРИЧИНА 5 ИСПРАВЛЕНА: Убрали запятые (,) из сплиттера, чтобы TTS не заикался на коротких кусках
const SENTENCE_BOUNDARY_REGEX = /([.?!]+(?:\s+|$))/;

class SentenceSplitter {
  private buffer = '';

  private findSoftBoundary() {
    const compactBuffer = this.buffer.replace(/\s+/g, ' ').trim();
    if (compactBuffer.length < SOFT_SPLIT_TARGET_CHARS) return -1;

    const punctuationCandidates = [', ', '; ', ': ', ' - ', ' and ', ' but ', ' because ', ' so '];
    let boundary = -1;

    for (const candidate of punctuationCandidates) {
      const index = this.buffer.lastIndexOf(candidate, SOFT_SPLIT_TARGET_CHARS);
      if (index > boundary) {
        boundary = index + candidate.length;
      }
    }

    if (boundary < SOFT_SPLIT_MIN_CHARS) {
      const fallbackIndex = this.buffer.lastIndexOf(' ', SOFT_SPLIT_TARGET_CHARS);
      if (fallbackIndex >= SOFT_SPLIT_MIN_CHARS) {
        boundary = fallbackIndex;
      }
    }

    return boundary;
  }

  processChunk(chunk: string, onSentenceReady: (sentence: string) => void) {
    this.buffer += chunk;
    
    while (true) {
      const match = this.buffer.match(SENTENCE_BOUNDARY_REGEX);
      if (match && match.index !== undefined) {
        const splitIndex = match.index + match[0].length;
        const sentence = this.buffer.substring(0, splitIndex).trim();
        
        if (sentence) {
          onSentenceReady(sentence);
        }
        
        this.buffer = this.buffer.substring(splitIndex);
      } else {
        const softBoundary = this.findSoftBoundary();
        if (softBoundary !== -1) {
          const sentence = this.buffer.substring(0, softBoundary).trim();
          if (sentence) {
            onSentenceReady(sentence);
          }
          this.buffer = this.buffer.substring(softBoundary).trimStart();
        } else {
          break;
        }
      }
    }
  }

  flush(onSentenceReady: (sentence: string) => void) {
    const sentence = this.buffer.trim();
    if (sentence) {
      onSentenceReady(sentence);
    }
    this.buffer = '';
  }
}

function extractSseEvents(buffer: string) {
  const normalizedBuffer = buffer.replace(/\r\n/g, '\n');
  const events: string[] = [];
  let searchIndex = normalizedBuffer.indexOf('\n\n');
  let lastIndex = 0;

  while (searchIndex !== -1) {
    events.push(normalizedBuffer.slice(lastIndex, searchIndex));
    lastIndex = searchIndex + 2;
    searchIndex = normalizedBuffer.indexOf('\n\n', lastIndex);
  }

  return {
    events,
    remaining: normalizedBuffer.slice(lastIndex),
  };
}

function parseSseEvent(eventBlock: string) {
  const dataLines = eventBlock
    .split('\n')
    .filter(line => line.startsWith('data:'))
    .map(line => line.slice(5).trimStart())
    .filter(Boolean);

  if (dataLines.length === 0) {
    return null;
  }

  const dataStr = dataLines.join('\n');
  if (dataStr === '[DONE]') {
    return null;
  }

  try {
    return JSON.parse(dataStr);
  } catch (error) {
    console.warn('Error parsing SSE event:', error, dataStr);
    return null;
  }
}

// ==========================================
// 3. Gemini API Integration (Streaming)
// ==========================================
async function generateResponse(userText: string, responseId: number) {
  // Update history with user's message
  state.messages.push({ role: 'user', parts: [{ text: userText }] });
  compressHistoryIfNeeded();

  try {
    const lessonMemory = state.lessonMemory.join(' | ');
    const inputTokens =
      estimateTokens(lessonMemory) +
      state.messages.reduce((total, message) => total + estimateTokens(getMessageText(message)), 0);

    state.estimatedInputTokens += inputTokens;
    publishSessionMetrics();

    const response = await fetch('/api/llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: state.messages,
          tutorMode: state.tutorMode,
          lessonFocus: state.lessonFocus,
          lessonMemory,
        })
      });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    if (!response.body) throw new Error('No response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    const splitter = new SentenceSplitter();
    
    let aiFullResponse = '';

    let sseBuffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      if (state.isAborted) {
        await reader.cancel();
        rollbackPendingUserTurn();
        return;
      }

      sseBuffer += decoder.decode(value, { stream: true });
      const { events, remaining } = extractSseEvents(sseBuffer);
      sseBuffer = remaining;

      for (const eventBlock of events) {
        if (state.isAborted) {
          await reader.cancel();
          rollbackPendingUserTurn();
          return;
        }

        const data = parseSseEvent(eventBlock);
        const textChunk = data?.candidates?.[0]?.content?.parts?.[0]?.text || '';

        if (textChunk) {
          aiFullResponse += textChunk;
          self.postMessage({ type: 'TEXT_CHUNK', payload: { text: textChunk, responseId } });

          splitter.processChunk(textChunk, (sentence) => {
            self.postMessage({ type: 'SENTENCE_READY', payload: { text: sentence, responseId } });
          });
        }
      }
    }

    sseBuffer += decoder.decode();
    const finalEvents = extractSseEvents(sseBuffer);
    const trailingEvents = finalEvents.remaining.trim() ? [...finalEvents.events, finalEvents.remaining] : finalEvents.events;

    for (const eventBlock of trailingEvents) {
      const data = parseSseEvent(eventBlock);
      const textChunk = data?.candidates?.[0]?.content?.parts?.[0]?.text || '';

      if (textChunk) {
        aiFullResponse += textChunk;
        self.postMessage({ type: 'TEXT_CHUNK', payload: { text: textChunk, responseId } });
        splitter.processChunk(textChunk, (sentence) => {
          self.postMessage({ type: 'SENTENCE_READY', payload: { text: sentence, responseId } });
        });
      }
    }

    // Flush remaining text in the buffer
    splitter.flush((sentence) => {
      self.postMessage({ type: 'SENTENCE_READY', payload: { text: sentence, responseId } });
    });

    // Update history with AI's full response
    state.messages.push({ role: 'model', parts: [{ text: aiFullResponse }] });
    state.estimatedOutputTokens += estimateTokens(aiFullResponse);
    compressHistoryIfNeeded();
    publishSessionMetrics();

    self.postMessage({ type: 'STREAM_END', payload: { responseId } });

  } catch (error) {
    rollbackPendingUserTurn();
    publishSessionMetrics();
    console.error('Gemini API Error:', error);
    self.postMessage({
      type: 'ERROR',
      payload: {
        responseId,
        message: error instanceof Error ? error.message : String(error),
      }
    });
  }
}

// ==========================================
// 4. Message Event Listener
// ==========================================
self.addEventListener('message', async (event: MessageEvent) => {
  const { type, payload } = event.data;
  
  switch (type) {
    case 'INIT_LLM':
      // Backend handles keys now, so we just say we're ready
      self.postMessage({ type: 'LLM_READY' });
      publishSessionMetrics();
      break;
      
    case 'ABORT_GENERATION':
      state.isAborted = true;
      break;

    case 'SET_TUTOR_MODE':
      if (payload.mode) {
        compressHistoryIfNeeded();
        const nextMode = payload.mode as TutorMode;
        state.tutorMode = nextMode;
        state.lessonFocus = payload.lessonFocus || DEFAULT_FOCUS[nextMode];
      }
      publishSessionMetrics();
      break;

    case 'SET_LESSON_FOCUS':
      state.lessonFocus = payload.lessonFocus || DEFAULT_FOCUS[state.tutorMode];
      publishSessionMetrics();
      break;

    case 'COMPRESS_HISTORY':
      compressHistoryIfNeeded(true);
      publishSessionMetrics();
      break;

    case 'RESET_SESSION_BUDGET':
      state.messages = [];
      state.lessonMemory = [];
      state.estimatedInputTokens = 0;
      state.estimatedOutputTokens = 0;
      state.lessonFocus = DEFAULT_FOCUS[state.tutorMode];
      publishSessionMetrics();
      break;

    case 'GENERATE_RESPONSE':
      state.isAborted = false;
      console.log('Generating response for:', payload.prompt);
      await generateResponse(payload.prompt, payload.responseId);
      break;
      
    default:
      console.error('Unknown message type:', type);
  }
});
