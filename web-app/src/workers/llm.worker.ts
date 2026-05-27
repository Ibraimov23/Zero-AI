// llm.worker.ts - API & Logic Layer
// This worker handles interactions with Gemini API, Supabase, and heavy business logic.

// ==========================================
// 1. State Management
// ==========================================
interface MessagePart {
  text: string;
}

interface ChatMessage {
  role: 'user' | 'model';
  parts: MessagePart[];
}

const state = {
  messages: [] as ChatMessage[],
  apiKey: import.meta.env.VITE_GEMINI_API_KEY || '',
  isAborted: false,
};

// ==========================================
// NEW: Sliding Window Context Management
// Prevents context overflow and memory leaks
// ==========================================
const MAX_CONTEXT_MESSAGES = 10;

function trimContextWindow() {
  // Keep the System Instruction (index 0)
  // If we exceed the limit, remove the oldest User/Model pair (index 1 and 2)
  if (state.messages.length > MAX_CONTEXT_MESSAGES + 1) {
    state.messages.splice(1, 2);
  }
}

// System instruction for the English Mentor persona
const SYSTEM_INSTRUCTION = {
  parts: [
    {
      text: `You are Zero AI, a highly advanced, concise, and conversational voice assistant. 
RULES:
1. Keep answers EXTREMELY short (1-2 sentences maximum).
2. Never use lists, bullet points, or markdown formatting.
3. Speak naturally like a human in a fast-paced dialogue.
4. If the user asks a quick question, give a quick answer.`
    }
  ]
};

// ==========================================
// 2. Sentence Splitter (Buffering)
// ==========================================
// Matches sentence endings: ., ?, !, followed by a space or end of string
// 🛑 ПРИЧИНА 5 ИСПРАВЛЕНА: Убрали запятые (,) из сплиттера, чтобы TTS не заикался на коротких кусках
const SENTENCE_BOUNDARY_REGEX = /([.?!]+(?:\s+|$))/;

class SentenceSplitter {
  private buffer = '';

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
        break;
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

// ==========================================
// 3. Gemini API Integration (Streaming)
// ==========================================
async function generateResponse(userText: string) {
  if (!state.apiKey) {
    self.postMessage({ type: 'ERROR', payload: 'Gemini API key is missing. Set VITE_GEMINI_API_KEY in .env' });
    return;
  }

  // Update history with user's message
  state.messages.push({ role: 'user', parts: [{ text: userText }] });

  // Trim context to prevent overflow
  trimContextWindow();

  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=${state.apiKey}`;
    
    const requestBody = {
      systemInstruction: SYSTEM_INSTRUCTION,
      contents: state.messages,
      generationConfig: {
        maxOutputTokens: 150,
        temperature: 0.7,
      }
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    if (!response.body) throw new Error('No response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    const splitter = new SentenceSplitter();
    
    let aiFullResponse = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
          if (state.isAborted) {
            reader.cancel();
            return;
          }

          if (line.startsWith('data: ')) {
          const dataStr = line.substring(6);
          if (dataStr === '[DONE]') continue;
          
          try {
            const data = JSON.parse(dataStr);
            const textChunk = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
            
            if (textChunk) {
              aiFullResponse += textChunk;
              // Send raw chunk to UI for typewriter effect
              self.postMessage({ type: 'TEXT_CHUNK', payload: textChunk });
              
              // Process through sentence splitter for TTS
              splitter.processChunk(textChunk, (sentence) => {
                self.postMessage({ type: 'SENTENCE_READY', payload: sentence });
              });
            }
          } catch (e) {
            console.warn('Error parsing SSE data line:', e);
          }
        }
      }
    }

    // Flush remaining text in the buffer
    splitter.flush((sentence) => {
      self.postMessage({ type: 'SENTENCE_READY', payload: sentence });
    });

    // Update history with AI's full response
    state.messages.push({ role: 'model', parts: [{ text: aiFullResponse }] });

    self.postMessage({ type: 'STREAM_END' });

  } catch (error) {
    console.error('Gemini API Error:', error);
    self.postMessage({ type: 'ERROR', payload: error instanceof Error ? error.message : String(error) });
  }
}

// ==========================================
// 4. Message Event Listener
// ==========================================
self.addEventListener('message', async (event: MessageEvent) => {
  const { type, payload } = event.data;
  
  switch (type) {
    case 'INIT_LLM':
      console.log('Initializing LLM Worker...');
      if (payload?.apiKey) {
        state.apiKey = payload.apiKey;
      }
      self.postMessage({ type: 'LLM_READY' });
      break;
      
    case 'ABORT_GENERATION':
      state.isAborted = true;
      break;

    case 'GENERATE_RESPONSE':
      state.isAborted = false;
      console.log('Generating response for:', payload.prompt);
      await generateResponse(payload.prompt);
      break;
      
    default:
      console.error('Unknown message type:', type);
  }
});
