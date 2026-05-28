/// <reference types="node" />

export const config = { runtime: 'edge' };

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse';
const RETRYABLE_STATUS_CODES = new Set([429, 500, 502, 503, 504]);
const MAX_RETRIES = 2;
const RETRY_BASE_DELAY_MS = 220;

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export default async function handler(req: Request) {
  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: 'Gemini API key is not configured on the server.' }), { status: 500 });
  }

  try {
    // Read the incoming JSON body
    const body = await req.json();
    const tutorMode = body.tutorMode || 'free_speaking';
    const lessonFocus = typeof body.lessonFocus === 'string' ? body.lessonFocus.trim() : '';
    const lessonMemory = typeof body.lessonMemory === 'string' ? body.lessonMemory.trim() : '';

    const modeInstruction =
      tutorMode === 'grammar'
        ? 'Prioritize grammar accuracy. Give a short correction, then one natural version.'
        : 'Prioritize natural free speaking. Sound current, relaxed, and easy to reply to.';

    const memoryInstruction = lessonMemory
      ? `Use memory only if it helps now: ${lessonMemory}.`
      : 'No useful lesson memory yet.';
    const focusInstruction = lessonFocus
      ? `Current lesson focus: ${lessonFocus}. Prioritize it when replying.`
      : 'Current lesson focus: general conversation.';

    // Re-inject the system instruction on the server side for security
    const requestBody = {
      systemInstruction: {
        parts: [{ text: `You are Zero AI, a smart English voice tutor with a modern playful vibe. Speak in clear natural spoken English. Sound witty, current, and lightly funny like a cool teen or young adult, but never cringe, rude, chaotic, or overloaded with slang. Use modern phrases naturally and sparingly. Stay concise, always finish the current thought, never rush, give only short useful corrections, ask at most one short follow-up question when it helps, and never use lists or markdown. In grammar mode, accuracy comes first. In free speaking mode, keep the conversation fun, smooth, and easy to continue. ${modeInstruction} ${focusInstruction} ${memoryInstruction}` }]
      },
      contents: body.contents,
      generationConfig: {
        temperature: 0.55, // Slightly more playful and natural without getting unstable
        maxOutputTokens: 240, // Enough to finish a thought calmly with a bit more personality
        topK: 1, // Faster sampling
        stopSequences: ["\n\n", "User:"], // Stops generation immediately if it tries to hallucinate a dialogue or write paragraphs
      }
    };

    // Retry only temporary upstream failures so users do not see raw provider flakiness.
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      const response = await fetch(`${GEMINI_API_URL}&key=${apiKey}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: req.signal,
      });

      if (response.ok) {
        return new Response(response.body, {
          status: response.status,
          headers: response.headers,
        });
      }

      const retryable = RETRYABLE_STATUS_CODES.has(response.status);
      if (retryable && attempt < MAX_RETRIES) {
        await wait(RETRY_BASE_DELAY_MS * 2 ** attempt);
        continue;
      }

      const errorText = (await response.text()).trim();
      return new Response(
        JSON.stringify({
          error: retryable
            ? 'Gemini is temporarily busy. Please try again.'
            : `Gemini upstream error: ${response.status} ${response.statusText}`,
          retryable,
          status: response.status,
          details: errorText.slice(0, 500),
        }),
        {
          status: response.status,
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-store',
          },
        }
      );
    }

    return new Response(JSON.stringify({ error: 'Gemini request failed after retries.', retryable: true }), {
      status: 503,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
      },
    });
  } catch (error: any) {
    const isAbortError = error?.name === 'AbortError';
    return new Response(
      JSON.stringify({
        error: isAbortError ? 'Gemini request was cancelled.' : error.message,
        retryable: !isAbortError,
      }),
      {
        status: isAbortError ? 499 : 500,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store',
        },
      }
    );
  }
}
