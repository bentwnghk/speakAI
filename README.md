# Mr.🆖 SpeakAI

**Turn text into natural-sounding audio with karaoke-style playback.** Upload documents, paste text, or extract text from images — then listen with word-by-word highlighting, adjustable speed, and multiple voices.

## Features

- **Multi-format input** — Paste text, upload PDF/DOCX/TXT files, or extract text from images via AI vision (JPG, PNG)
- **6 high-quality voices** — Three female (Nova, Phoebe, Ava) and three male (Alloy, Adam, Ollie) voices powered by Azure Speech
- **Karaoke playback** — Word-by-word highlighting synced to the audio, using ground-truth timing from the synthesis engine
- **Speed control** — Adjustable from 0.5x to 2.0x
- **Credit system** — Pay-as-you-go with Stripe integration (1 credit = HK$1); new users get 3 free credits
- **Generation history** — Browse, rename, and replay past generations; audio files served with Range support for seeking
- **PWA** — Installable as a standalone app with offline support via Serwist service worker
- **Dark mode** — System/Light/Dark themes with FOUC prevention and cross-device sync
- **i18n** — English and Traditional Chinese (繁體中文) with type-safe translations
- **Multi-endpoint load balancing** — Round-robin across multiple Azure Speech endpoints for even cost distribution

## Tech Stack

| Layer | Technology |
| --- | --- |
| Framework | Next.js 15 (App Router, React 19, Server Components) |
| Language | TypeScript (strict mode) |
| Styling | Tailwind CSS v4 (OKLCH color variables) |
| UI Components | Shadcn UI (new-york style) + Radix primitives |
| Auth | NextAuth v5 (Google OAuth, JWT strategy) |
| Database | PostgreSQL 16 + Drizzle ORM |
| TTS | Azure Speech SDK with word-boundary alignment |
| Payments | Stripe Checkout + webhooks |
| PWA | Serwist (service worker generation) |
| Animations | Motion (Framer Motion successor) |
| Validation | Zod v3 |
| Container | Docker (multi-stage, node:22-slim, non-root) |
| CI/CD | GitHub Actions → Docker Hub + GHCR (amd64/arm64) |

## Getting Started

### Prerequisites

- Node.js >= 18
- PostgreSQL 16
- Azure Speech resource
- Google OAuth credentials
- Stripe account (for payments)

### Local Development

1. **Clone and install:**

   ```bash
   git clone https://github.com/bentwnghk/speakAI.git
   cd speakAI
   npm install
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env.local
   ```

   Edit `.env.local` with your credentials (see [Configuration](#configuration)).

3. **Set up the database:**

   ```bash
   npm run db:push
   ```

4. **Start the dev server:**

   ```bash
   npm run dev
   ```

   Open [http://localhost:3000](http://localhost:3000).

### Docker

```bash
docker compose up --build
```

This starts PostgreSQL on port 5432 and the app on port 3000. Audio data persists in a Docker volume.

## Configuration

All configuration is via environment variables. Copy `.env.example` as a starting point.

### Required

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string |
| `AUTH_SECRET` | NextAuth secret key |
| `AUTH_GOOGLE_ID` | Google OAuth client ID |
| `AUTH_GOOGLE_SECRET` | Google OAuth client secret |
| `AZURE_SPEECH_KEY_1` | Azure Speech subscription key |
| `AZURE_SPEECH_REGION_1` | Azure Speech region |

### Optional

| Variable | Default | Purpose |
| --- | --- | --- |
| `AZURE_SPEECH_KEY_2`…`_20` | — | Additional Azure endpoints for round-robin load balancing |
| `AZURE_SPEECH_PRICE_USD_PER_1M_CHARS` | `16` | Cost per 1M characters (USD) |
| `VISION_API_KEY` | falls back to `TTS_API_KEY` | API key for image OCR |
| `VISION_BASE_URL` | falls back to `TTS_BASE_URL` | Base URL for vision model |
| `VISION_MODEL` | `gpt-4.1-mini` | Vision model name |
| `STRIPE_SECRET_KEY` | — | Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | — | Stripe webhook signing secret |
| `WELCOME_CREDITS` | `3` | Free credits for new users |
| `STRIPE_PLAN_A_CREDITS` / `_PRICE_HKD` | 15 / 15 | Starter plan |
| `STRIPE_PLAN_B_CREDITS` / `_PRICE_HKD` | 50 / 45 | Best Value plan |
| `AUDIO_RETENTION_DAYS` | `365` | Days before auto-deletion of audio files |

## Project Structure

```
src/
├── app/
│   ├── (auth)/              # Login page (unauthenticated)
│   ├── (dashboard)/         # Main app (authenticated)
│   │   ├── page.tsx         # TTS generation page
│   │   ├── credits/         # Credit purchase page
│   │   └── history/         # Generation history
│   └── api/                 # API route handlers
├── components/
│   ├── tts-form.tsx         # Main TTS form
│   ├── audio-player.tsx     # Audio player with karaoke
│   ├── voice-select.tsx     # Voice selector
│   ├── landing/             # Public landing page
│   └── ui/                  # Shadcn UI primitives
├── lib/
│   ├── auth.ts              # NextAuth v5 config
│   ├── tts.ts               # Azure Speech TTS + karaoke alignment
│   ├── stripe.ts            # Stripe client + plan definitions
│   ├── i18n/                # Type-safe i18n (en, zh-TW)
│   └── db/
│       ├── schema.ts        # 9 tables (Drizzle ORM)
│       └── credits.ts       # Credit engine
├── hooks/
│   ├── use-credits.tsx      # Credit balance context
│   └── use-settings.tsx     # Theme + locale context
└── sw/
    └── index.ts             # Serwist service worker
```

## API Endpoints

| Endpoint | Methods | Description |
| --- | --- | --- |
| `/api/tts` | POST | Generate TTS audio (deducts credits) |
| `/api/tts` | GET | List user's generations |
| `/api/extract-text` | POST | Extract text from uploaded file |
| `/api/generations/[id]` | GET / PATCH / DELETE | Single generation CRUD |
| `/api/audio/[id]` | GET | Serve audio (Range support) |
| `/api/stripe/checkout` | POST | Create Stripe Checkout Session |
| `/api/stripe/webhook` | POST | Stripe webhook handler |
| `/api/stripe/plans` | GET | Available credit plans |
| `/api/user/credits` | GET | Credit balance |
| `/api/user/settings` | GET / PUT | Theme and locale preferences |

All endpoints except `/api/auth/*` and `/api/stripe/webhook` require authentication.

## Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start development server |
| `npm run build` | Production build |
| `npm run start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm run db:generate` | Generate Drizzle migrations |
| `npm run db:push` | Push schema to database |
| `npm run db:migrate` | Run migrations |
| `npm run db:studio` | Open Drizzle Studio |

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

[Apache 2.0](LICENSE)
