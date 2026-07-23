import { Serwist } from "serwist";
import type { PrecacheEntry } from "serwist";

interface SwManifest {
  __SW_MANIFEST: Array<PrecacheEntry | string>;
}

const swSelf = self as unknown as SwManifest;

const serwist = new Serwist({
  precacheEntries: swSelf.__SW_MANIFEST,
  skipWaiting: true,
  clientsClaim: true,
});

serwist.addEventListeners();
