/// <reference types="node" />

export const config = { runtime: 'edge' };

export default async function handler(req: Request) {
  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: 'OpenAI API key is not configured on the server.' }), { status: 500 });
  }

  try {
    const sdp = await req.text();
    if (!sdp.trim()) {
      return new Response(JSON.stringify({ error: 'Missing SDP offer.' }), { status: 400 });
    }

    const session = {
      type: 'transcription',
      audio: {
        input: {
          noise_reduction: { type: 'near_field' },
          transcription: {
            model: 'gpt-4o-mini-transcribe',
            language: 'en',
            prompt: 'Natural English tutoring conversation. Ignore silence, filler-only sounds, and random background noise.',
          },
          turn_detection: {
            type: 'server_vad',
            threshold: 0.55,
            prefix_padding_ms: 300,
            silence_duration_ms: 650,
          },
        },
      },
    };

    const formData = new FormData();
    formData.set('sdp', sdp);
    formData.set('session', JSON.stringify(session));

    const response = await fetch('https://api.openai.com/v1/realtime/calls', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      body: formData,
    });

    const answerSdp = await response.text();
    return new Response(answerSdp, {
      status: response.status,
      headers: {
        'Content-Type': 'application/sdp',
      },
    });
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), { status: 500 });
  }
}
