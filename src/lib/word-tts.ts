import * as sdk from "microsoft-cognitiveservices-speech-sdk";

const DEFAULT_VOICE = "en-US-NovaTurboMultilingualNeural";

interface AzureEndpoint {
  key: string;
  region: string;
}

function loadAzureEndpoints(): AzureEndpoint[] {
  const endpoints: AzureEndpoint[] = [];
  for (let i = 1; i <= 20; i++) {
    const key = process.env[`AZURE_SPEECH_KEY_${i}`];
    const region = process.env[`AZURE_SPEECH_REGION_${i}`];
    if (key && region) endpoints.push({ key, region });
  }
  if (endpoints.length === 0) {
    const key = process.env.AZURE_SPEECH_KEY;
    const region = process.env.AZURE_SPEECH_REGION;
    if (key && region) endpoints.push({ key, region });
  }
  return endpoints;
}

const AZURE_ENDPOINTS = loadAzureEndpoints();
let rrCursor = 0;

function pickNextEndpoint(): AzureEndpoint {
  if (AZURE_ENDPOINTS.length === 0) {
    throw new Error("No Azure Speech endpoints configured.");
  }
  const ep = AZURE_ENDPOINTS[rrCursor % AZURE_ENDPOINTS.length];
  rrCursor++;
  return ep;
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export async function synthesizeWord(word: string): Promise<ArrayBuffer> {
  const { key, region } = pickNextEndpoint();

  const speechConfig = sdk.SpeechConfig.fromSubscription(key, region);
  speechConfig.speechSynthesisVoiceName = DEFAULT_VOICE;
  speechConfig.speechSynthesisOutputFormat =
    sdk.SpeechSynthesisOutputFormat.Audio24Khz96KBitRateMonoMp3;

  const synthesizer = new sdk.SpeechSynthesizer(speechConfig, null);

  const ssml = `<speak version="1.0" xml:lang="en-US"><voice name="${DEFAULT_VOICE}"><prosody rate="-15%">${escapeXml(word)}</prosody></voice></speak>`;

  try {
    return await new Promise<ArrayBuffer>((resolve, reject) => {
      synthesizer.speakSsmlAsync(
        ssml,
        (result) => {
          synthesizer.close();
          if (result.reason !== sdk.ResultReason.SynthesizingAudioCompleted) {
            reject(new Error(result.errorDetails || "Azure Speech synthesis failed"));
            return;
          }
          resolve(result.audioData);
        },
        (error) => {
          synthesizer.close();
          reject(new Error(error));
        },
      );
    });
  } catch (error) {
    synthesizer.close();
    throw error;
  }
}
