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

    // Re-inject the system instruction on the server side for security
    const requestBody = {
      systemInstruction: {
        parts: [{ text: "You are Zero AI, a highly advanced, concise, and conversational voice assistant. RULES: 1. Keep answers EXTREMELY short (1-2 sentences maximum). 2. Never use lists, bullet points, or markdown formatting. 3. Speak naturally like a human in a fast-paced dialogue. 4. If the user asks a quick question, give a quick answer." }]
      },
      contents: body.contents,
      generationConfig: {
        temperature: 0.5, // Lower temperature = faster, more deterministic responses
        maxOutputTokens: 60, // STRICT COST CONTROL: Max ~40-50 words. Prevents run-on generations and saves money.
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