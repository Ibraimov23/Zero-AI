/// <reference types="node" />

export const config = { runtime: 'edge' };

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

    // Forward the request body directly to Gemini (SSE Stream)
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=${apiKey}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    return new Response(response.body, {
      status: response.status,
      headers: response.headers,
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), { status: 500 });
  }
}
