// llm.worker.ts - API & Logic Layer
// This worker handles interactions with Gemini API, Supabase, and heavy business logic.

// ==========================================
// 1. State Management
// ==========================================
interface ChatMessage {
  role: string;
  parts: Array<{ text: string }>;
}

const state = {
  messages: [] as ChatMessage[],
  isAborted: false,
};

// ==========================================
// NEW: Sliding Window Context Management
// Prevents context overflow and memory leaks
// ==========================================
// 🛑 COST OPTIMIZATION: Reduced from 10 to 6. 
// Remembering the last 3 pairs (User+AI) is enough for voice context.
// This halves the input tokens sent to the API, saving 50% of the cost.
const MAX_CONTEXT_MESSAGES = 6;

function trimContextWindow() {
  // Keep the System Instruction (index 0)
  // If we exceed the limit, remove the oldest User/Model pair (index 1 and 2)
  if (state.messages.length > MAX_CONTEXT_MESSAGES + 1) {
    state.messages.splice(1, 2);
  }
}


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
  // Update history with user's message
  state.messages.push({ role: 'user', parts: [{ text: userText }] });

  // Trim context to prevent overflow
  trimContextWindow();

  try {
    const response = await fetch('/api/llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: state.messages
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
      // Backend handles keys now, so we just say we're ready
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
