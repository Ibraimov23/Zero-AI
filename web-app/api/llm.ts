/// <reference types="node" />

export const config = { runtime: 'edge' };

export default async function handler(req: Request) {
  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 });
  }

  const apiKey = process.env.GEMINI_API_KEY || process.env.VITE_GEMINI_API_KEY;
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
        ? 'Focus on grammar accuracy first. Give short corrections, then a natural corrected version.'
        : 'Focus on natural free speaking. Keep the learner talking with calm conversational practice.';

    const memoryInstruction = lessonMemory
      ? `Lesson memory: ${lessonMemory}. Use it only if it helps the learner immediately.`
      : 'Lesson memory: none yet.';
    const focusInstruction = lessonFocus
      ? `Current lesson focus: ${lessonFocus}. Prioritize it when replying.`
      : 'Current lesson focus: general conversation.';

    // Re-inject the system instruction on the server side for security
    const requestBody = {
      systemInstruction: {
        parts: [{ text: `You are Zero AI, a calm and natural English tutor for voice conversations. RULES: 1. Reply in clear spoken English that is easy for a learner to follow. 2. Keep answers concise, but always finish the thought naturally before stopping. 3. Do not rush the conversation or jump to a new topic before closing the current reply. 4. Correct mistakes gently only when it helps learning, and keep the correction very short. 5. Ask at most one short follow-up question, and only when it naturally helps continue practice. 6. Prefer a warm phone-call rhythm: calm, clear, and human. 7. Never use lists, bullet points, or markdown formatting. 8. ${modeInstruction} 9. ${focusInstruction} 10. ${memoryInstruction}` }]
      },
      contents: body.contents,
      generationConfig: {
        temperature: 0.4, // More stable and tutor-like, with fewer rambly responses
        maxOutputTokens: 220, // Enough to finish a thought calmly without wasting tokens
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
