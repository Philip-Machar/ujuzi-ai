import 'dotenv/config';
import express from 'express';
import crypto from 'crypto';
import Redis from 'ioredis';
import OpenAI from 'openai';

// ---------------------------------------------------------------------------
// 1. INITIALIZE CLIENTS (REDIS + META MODEL API)
// ---------------------------------------------------------------------------
const app = express();
const redis = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');

const metaClient = new OpenAI({
  baseURL: 'https://api.meta.ai/v1',
  apiKey: process.env.MODEL_API_KEY,
});

const META_MODEL = process.env.META_MODEL_ID || 'muse-spark-1.3';
const HISTORY_TTL_SECONDS = 1800; // 30 minutes of conversation memory
const IDEMPOTENCY_TTL_SECONDS = 86400; // 24 hours duplicate retry protection
const MAX_HISTORY_MESSAGES = 6; // Last 3 user + 3 assistant turns

// ---------------------------------------------------------------------------
// 2. RAW-BYTE JSON PARSER (GUARDRAIL #3: EXACT HMAC VERIFICATION)
// ---------------------------------------------------------------------------
app.use(
  express.json({
    verify: (req, _res, buf) => {
      req.rawBody = buf; // Preserve exact bytes for HMAC-SHA256 verification
    },
  })
);

// ---------------------------------------------------------------------------
// 3. HELPER FUNCTIONS: SECURITY, GSM-7 SANITIZATION & SMS DISPATCH
// ---------------------------------------------------------------------------

/**
 * Verifies TextBee's X-Signature header against the raw request buffer.
 */
function verifyWebhookSignature(rawBodyBuffer, signatureHeader, secret) {
  if (!secret) return true; // Bypass only if secret is not configured in local dev
  if (!rawBodyBuffer || !signatureHeader) return false;

  const cleanSig = signatureHeader.replace(/^sha256=/i, '').trim();
  const expectedHex = crypto
    .createHmac('sha256', secret)
    .update(rawBodyBuffer)
    .digest('hex');

  const sigBuffer = Buffer.from(cleanSig, 'utf8');
  const expectedBuffer = Buffer.from(expectedHex, 'utf8');

  if (sigBuffer.length !== expectedBuffer.length) return false;
  return crypto.timingSafeEqual(sigBuffer, expectedBuffer);
}

/**
 * Guardrail #4: Only allow valid E.164 mobile numbers (blocks "MPESA", "Safaricom", etc.)
 */
function isValidMobileE164(sender) {
  return typeof sender === 'string' && /^\+[1-9]\d{9,14}$/.test(sender.trim());
}

/**
 * Guardrail #2: Converts Unicode punctuation to plain GSM-7 ASCII and enforces <=155 chars
 * so the mobile carrier never drops from 160 chars/SMS down to 70 chars/SMS (UCS-2).
 */
function sanitizeToGsm7(text, maxLength = 155) {
  if (!text) return 'Samahani, jaribu tena baada ya muda mfupi.';

  let clean = text
    // Normalize accented characters to base ASCII
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    // Replace smart/curly quotes and apostrophes with straight ASCII
    .replace(/[\u2018\u2019\u201A\u201B\u2032\u2035`]/g, "'")
    .replace(/[\u201C\u201D\u201E\u201F\u2033\u2036]/g, '"')
    // Replace em-dashes, en-dashes, and bullets with standard hyphen
    .replace(/[\u2013\u2014\u2015\u2022\u25CF\u25AA]/g, '-')
    // Replace ellipsis character with three dots
    .replace(/\u2026/g, '...')
    // Strip markdown bold/italics/headers
    .replace(/[*_~`#]/g, '')
    // Collapse newlines and multiple spaces into a single space
    .replace(/\s+/g, ' ')
    // Strip any remaining non-printable ASCII or emojis (keep ASCII 32-126)
    .replace(/[^\x20-\x7E]/g, '')
    .trim();

  if (clean.length > maxLength) {
    // Truncate cleanly at the last space before the limit
    const truncated = clean.slice(0, maxLength - 3);
    const lastSpace = truncated.lastIndexOf(' ');
    clean = (lastSpace > 80 ? truncated.slice(0, lastSpace) : truncated) + '...';
  }

  return clean;
}

/**
 * Sends the outbound SMS back through the TextBee Android Gateway.
 */
async function sendSmsViaTextBee(recipient, message) {
  const payload = {
    recipients: [recipient],
    message,
  };

  if (process.env.TEXTBEE_DEVICE_ID) {
    payload.deviceId = process.env.TEXTBEE_DEVICE_ID;
  }

  const response = await fetch('https://api.textbee.dev/api/v1/gateway/send-sms', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.TEXTBEE_API_KEY,
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`TextBee API error (${response.status}): ${errorText}`);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// 4. ASYNC BACKGROUND WORKER (REDIS MEMORY + MUSE SPARK 1.3 + SMS REPLY)
// ---------------------------------------------------------------------------
const DEVELOPER_PROMPT = `You are Ujuzi AI, an SMS assistant serving users in East Africa on basic feature phones.
STRICT RULES:
1. Reply in the EXACT language used by the user (Swahili, Sheng, English, or local vernacular).
2. For Swahili, use natural, clear East African Kiswahili.
3. Output PLAIN ASCII TEXT ONLY: no emojis, no markdown (* or #), no bullet lists, no newlines.
4. HARD CHARACTER LIMIT: Your entire response MUST be under 150 characters so it fits in 1 SMS.
Example User: Ninawezaje kutibu kiungulia?
Example Reply: Kunywa maji mengi, epuka vyakula vya mafuta au pilipili, na usilale mara moja baada ya kula. Kikiendelea zaidi ya siku 3, muone daktari.`;

async function processInboundSms({ sender, userMessage, uniqueEventKey }) {
  try {
    // A. Atomic Redis Idempotency Check (SET key 1 EX 86400 NX)
    const isFirstTime = await redis.set(
      `idem:${uniqueEventKey}`,
      '1',
      'EX',
      IDEMPOTENCY_TTL_SECONDS,
      'NX'
    );

    if (!isFirstTime) {
      console.log(`[Duplicate Ignored] Event ${uniqueEventKey} already processed.`);
      return;
    }

    // B. Fetch sliding-window conversation history from Redis
    const chatKey = `chat:${sender}`;
    const rawHistory = await redis.get(chatKey);
    const history = rawHistory ? JSON.parse(rawHistory) : [];

    // C. Build messages array for Muse Spark 1.3
    const messages = [
      { role: 'developer', content: DEVELOPER_PROMPT },
      ...history,
      { role: 'user', content: userMessage },
    ];

    // D. Call Meta Model API (muse-spark-1.3 with minimal reasoning overhead)
    const completion = await metaClient.chat.completions.create({
      model: META_MODEL,
      messages,
      reasoning_effort: 'minimal',
      temperature: 0.3,
      max_completion_tokens: 400,
    });

    const rawAiReply = completion.choices?.[0]?.message?.content || '';
    const smsSafeReply = sanitizeToGsm7(rawAiReply, 155);

    console.log(`[AI Reply -> ${sender}] (${smsSafeReply.length} chars): ${smsSafeReply}`);

    // E. Save updated 3-turn history back to Redis with 30-min sliding expiration
    const updatedHistory = [
      ...history,
      { role: 'user', content: userMessage },
      { role: 'assistant', content: smsSafeReply },
    ].slice(-MAX_HISTORY_MESSAGES);

    await redis.set(chatKey, JSON.stringify(updatedHistory), 'EX', HISTORY_TTL_SECONDS);

    // F. Dispatch SMS reply to the user's phone via TextBee
    await sendSmsViaTextBee(sender, smsSafeReply);
    console.log(`[SMS Queued Successfully] Sent to ${sender}`);
  } catch (error) {
    console.error(`[Worker Error for ${sender}]:`, error.message);
  }
}

// ---------------------------------------------------------------------------
// 5. WEBHOOK ROUTE (GUARDRAIL #1: IMMEDIATE 200 OK RESPONSE)
// ---------------------------------------------------------------------------
app.post('/webhooks/sms-inbound', (req, res) => {
  const signature = req.headers['x-signature'];

  // 1. Verify HMAC-SHA256 signature on raw bytes
  if (!verifyWebhookSignature(req.rawBody, signature, process.env.TEXTBEE_WEBHOOK_SECRET)) {
    console.warn('[Security] Invalid X-Signature rejected.');
    return res.status(401).json({ error: 'Invalid signature' });
  }

  const { smsId, sender, message, webhookEvent, idempotencyKey } = req.body || {};

  // 2. Acknowledge webhook immediately (<50ms) so TextBee never triggers retry loops
  res.status(200).json({ received: true });

  // 3. Filter out non-inbound events and carrier shortcodes (e.g., MPESA, Safaricom)
  if (webhookEvent && webhookEvent !== 'MESSAGE_RECEIVED') return;
  if (!isValidMobileE164(sender) || !message || !message.trim()) {
    console.log(`[Filtered] Ignoring non-user message from: ${sender}`);
    return;
  }

  const uniqueEventKey = idempotencyKey || smsId || `${sender}:${Date.now()}`;
  console.log(`[Inbound SMS <- ${sender}]: ${message.trim()}`);

  // 4. Fire-and-forget asynchronous processing
  setImmediate(() => {
    processInboundSms({
      sender: sender.trim(),
      userMessage: message.trim(),
      uniqueEventKey,
    });
  });
});

// Health-check endpoint for testing your ngrok tunnel
app.get('/health', (_req, res) => {
  res.status(200).json({ status: 'ok', service: 'Ujuzi AI SMS Gateway', model: META_MODEL });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Ujuzi AI Server running on http://localhost:${PORT}`);
  console.log(`Webhook endpoint ready at POST /webhooks/sms-inbound`);
});