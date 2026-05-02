This document provides essential guidelines and technical references for AI agents (and human developers) working on the **Mr.🆖 SpeakAI** repository. Adhere to these patterns to ensure consistency, security, and maintainability.

---

## Development Workflow & Commands

The project uses **npm** as the primary package manager (Node >= 18).

### Core Commands

- **Install Dependencies**: `npm install`
- **Development Server**: `npm run dev` (Runs at `http://localhost:3000`)
- **Build Project**: `npm run build`
- **Start Production**: `npm run start`
- **Linting**: `npm run lint`

### Database (Drizzle ORM)

- **Generate Migrations**: `npm run db:generate`
- **Push Schema**: `npm run db:push`
- **Run Migrations**: `npm run db:migrate`
- **Drizzle Studio**: `npm run db:studio`

### Testing

- **Status**: Currently, there are no automated tests in the codebase.
- **Guideline**: If adding tests, use **Vitest** following standard Next.js patterns. Place test files next to the code they test (e.g., `ComponentName.test.tsx`) or in a `__tests__` directory.

### Docker

- **Dockerfile**: Multi-stage build on `node:22-slim`, standalone output, runs as non-root `nextjs` user via `gosu`, exposes port 3000.
- **docker-compose.yml**: Two services — `db` (PostgreSQL 16 Alpine, port 5432, runs `scripts/init-db.sql` on init) and `app` (port 3000, depends on healthy db). Audio data persisted in `audio_data` volume.
- **Build & Run**: `docker compose up --build`

### CI/CD

- **`.github/workflows/docker-image.yml`**: Pushes multi-arch (amd64/arm64) Docker image to Docker Hub on `main` branch pushes.
- **`.github/workflows/ghcr.yml`**: Pushes Docker image to GitHub Container Registry on `main`/`dev` pushes and `v*` tags.

---

## Project Structure

```
src/
├── app/                        # Next.js App Router (Pages, API routes, Layouts)
│   ├── globals.css             # Tailwind v4 CSS-first config, OKLCH theme variables
│   ├── layout.tsx              # Root layout (AuthProvider, Toaster)
│   ├── manifest.ts             # PWA manifest (standalone, theme color)
│   ├── (auth)/                 # Unauthenticated route group
│   │   ├── layout.tsx          # Auth layout (redirects to / if session exists)
│   │   └── login/              # Login page
│   ├── (dashboard)/            # Authenticated route group (server-side auth guard in layout)
│   │   ├── layout.tsx          # Dashboard layout (Header, auth check, footer)
│   │   ├── page.tsx            # Main TTS generation page
│   │   ├── credits/            # Credit purchase page with Stripe
│   │   └── history/            # Generation history listing
│   └── api/                    # API route handlers (see Backend section)
├── components/
│   ├── auth-provider.tsx       # Client provider: SessionProvider + CreditsProvider wrapper
│   ├── audio-player.tsx        # Audio playback with karaoke-style highlighting
│   ├── file-upload.tsx         # File upload (PDF, DOCX, TXT, images)
│   ├── header.tsx              # App header with navigation + credits display
│   ├── history-list.tsx        # Generation history list component
│   ├── karaoke-text.tsx        # Word-by-word karaoke text display
│   ├── speed-slider.tsx        # Speed control slider
│   ├── tts-form.tsx            # Main TTS form (text input, voice, speed, file upload)
│   ├── voice-select.tsx        # Voice selection dropdown
│   ├── landing/                # Public landing/marketing page
│   │   └── landing-page.tsx
│   └── ui/                     # Shadcn UI primitives (do not modify directly)
├── lib/
│   ├── auth.ts                 # NextAuth v5 config (Google OAuth, JWT strategy, Drizzle adapter)
│   ├── constants.ts            # Voice mappings, supported file extensions
│   ├── utils.ts                # cn() utility (clsx + tailwind-merge)
│   ├── pdf-client.ts           # Client-side PDF processing (pdfjs-dist)
│   ├── file-parser.ts          # Server-side file text extraction (DOCX, TXT, images)
│   ├── tts.ts                  # TTS generation + Whisper-based audio alignment
│   ├── stripe.ts               # Stripe client singleton + credit plan definitions
│   └── db/
│       ├── index.ts            # Drizzle ORM setup (postgres-js driver, schema export)
│       ├── schema.ts           # 8 tables: user, account, session, verification_token, generation, credits, credit_transactions, purchases
│       └── credits.ts          # Credit engine: grant, deduct, refund, purchase operations
├── sw/
│   └── index.ts                # Serwist service worker for PWA offline support
├── hooks/
│   └── use-credits.tsx         # React context for credit balance (app-wide)
└── types/
    ├── karaoke.ts              # WordTimestamp, Segment types for karaoke display
    └── pdf-parse.d.ts          # Type declarations for pdf-parse
scripts/                        # SQL migrations (init-db.sql + incremental migrations)
```

---

## Code Style & Conventions

### 1. TypeScript & Types

- **Strict Mode**: `strict: true` is enabled in `tsconfig.json`. Always provide explicit types for function parameters and return values.
- **Global Types**: Core types (`WordTimestamp`, `Segment`) are defined in `src/types/karaoke.ts`. Check this file before creating new interfaces.
- **Zod**: Use **Zod v3** for schema validation where needed.

### 2. React & Next.js

- **App Router**: This project uses the Next.js App Router with route groups `(dashboard)` for authenticated and `(auth)` for unauthenticated pages.
- **Server Components**: Default to Server Components. Only add `"use client"` when browser APIs or React hooks (state, effects) are needed.
- **Route Groups**: `(dashboard)` has a server-side auth guard in its layout (calls `auth()`, redirects to `/login`). `(auth)` redirects authenticated users to `/`.
- **PWA**: Uses **Serwist** (`@serwist/next`) for service worker generation. Configured in `next.config.ts`, worker source in `src/sw/index.ts`.

### 3. Components & UI

- **Shadcn UI**: UI primitives are located in `@/components/ui`. Do not modify them directly; extend them or create wrappers. Style: `new-york`, base color: `neutral`.
- **Styling**: Uses **Tailwind CSS v4** with CSS-first configuration (no `tailwind.config.ts`). Theme uses OKLCH color variables. Follow mobile-first responsive design patterns.
- **Dark Mode**: Uses CSS `.dark` class with OKLCH variables. Ensure all new UI elements support both light and dark modes using Tailwind `dark:` classes.
- **Icons**: Use **lucide-react**.
- **Animations**: Use `motion` (Framer Motion successor).

### 4. State Management

- **Server State**: Next.js RSC with `auth()` calls for session data.
- **Client State**: Minimal — React hooks and local component state.
- **Provider Hierarchy**: `AuthProvider` wraps `SessionProvider` from `next-auth/react`.

### 5. Imports

- **Path Alias**: Always use the `@/` prefix for absolute imports from the `src` directory.
- **Ordering**:
  1. React/Next.js core
  2. Third-party libraries
  3. Components (Internal/UI)
  4. Lib & Utils
  5. Types

---

## Authentication (NextAuth v5)

The project uses **next-auth v5 (beta.25)** with **Google OAuth** as the sole provider.

- **`src/lib/auth.ts`**: Full config with `@auth/drizzle-adapter` (PostgreSQL). Session strategy is **JWT** (not database). Includes a custom `createUser` that checks for existing users by email before inserting.
- **Session**: JWT tokens carry `user.id` via `sub` claim. Client-side access via `AuthProvider` wrapping `SessionProvider`.
- **Route Protection**: Auth checks are performed in route group layouts, not via middleware. `(dashboard)/layout.tsx` calls `auth()` and redirects to `/login` if no session. `(auth)/layout.tsx` redirects to `/` if session exists.

---

## Credits & Payment System

### Overview

- **1 credit = HK$1.00** (stored as `REAL` with 2 decimal places).
- Each new user receives **3 free credits** on first sign-in (configurable via `WELCOME_CREDITS` env var).
- Each TTS generation consumes credits equal to the generation cost in HK$.
- Credits are deducted after successful generation; refunds are issued on failure.

### Credit Engine (`src/lib/db/credits.ts`)

- **`ensureCreditsRecord(userId)`**: Creates credits row with welcome bonus on first sign-in. Called from NextAuth `signIn` event.
- **`deductCredits(userId, amount, description)`**: Atomically decrements balance, records a `generation` transaction. Returns 402 if insufficient.
- **`refundCredits(userId, amount, description)`**: Increments balance, records a `refund` transaction.
- **`addCreditsFromPurchase(...)`**: Used by Stripe webhook to credit purchased amounts. Idempotent (upserts purchase record).

### Stripe Integration (`src/lib/stripe.ts`)

- **Plans**: 2 plans — Starter (HK$15 for 15 credits) and Best Value (HK$45 for 50 credits). Configurable via env vars.
- **Checkout flow**: POST `/api/stripe/checkout` → creates Stripe Checkout Session → inserts `pending` purchase → redirects to Stripe.
- **Webhook**: POST `/api/stripe/webhook` handles `checkout.session.completed` (credits user) and `checkout.session.expired` (marks failed).
- **No auth** on webhook route — Stripe signs requests with `STRIPE_WEBHOOK_SECRET`.

### Client-Side

- **`useCredits()` hook** (`src/hooks/use-credits.tsx`): React context providing `balance`, `loading`, `refreshBalance()`. Wrapped by `CreditsProvider` in `AuthProvider`.
- **Header**: Shows remaining credits (HK$ balance) as a button linking to `/credits`, right after History.
- **Credits page** (`/credits`): Displays balance, 2 plan cards with purchase buttons, success/cancel banners, and purchase history table.

---

## Database (PostgreSQL + Drizzle ORM)

The project uses **PostgreSQL 16** with **Drizzle ORM**.

- **Connection**: `postgres-js` driver configured in `src/lib/db/index.ts`.
- **Schema**: All tables defined in `src/lib/db/schema.ts` (8 tables: `user`, `account`, `session`, `verification_token`, `generation`, `credits`, `credit_transactions`, `purchases`). The `generation` table stores TTS generation history with voice (as a Postgres enum), speed, audio path, karaoke segments (JSON), and cost. The `credits` table stores per-user balance (1:1 with users). `credit_transactions` is an append-only ledger. `purchases` tracks Stripe payment lifecycle.
- **Migrations**: SQL migration files in `scripts/` (e.g., `init-db.sql`, `add-segments-column.sql`). Drizzle migrations in `drizzle/`.
- **Config**: `drizzle.config.ts` at project root.

---

## TTS & Audio Pipeline

### Text-to-Speech

- **`src/lib/tts.ts`**: Core TTS logic using OpenAI-compatible speech API (`tts-1` model).
- **Voices**: 6 voices available — nova, alloy, fable, echo, shimmer, onyx. Mapped from display names (e.g., "Female 1" → "nova") via `VOICE_MAP` in `src/lib/constants.ts`.
- **Text Chunking**: Long text is split into chunks (max 4000 chars) respecting paragraph and sentence boundaries. Chunks are processed in parallel (batch of 10).
- **Audio Output**: Generated MP3 files stored in `data/audio/` with nanoid-based filenames.

### Audio Alignment (Karaoke)

- **Whisper API**: After TTS generation, audio is sent to Whisper (`whisper-1`) with `verbose_json` format for both word and segment timestamps.
- **Word Alignment**: Source text is split into sentences, then mapped onto Whisper segments. Per-word timing is derived via character-position proportional alignment (`mapWordsToTimings`). This handles vocabulary mismatches between source and Whisper output.
- **Drift Correction**: Word timestamps are re-anchored to sentence boundaries (`normalizeWordTimingsToSegment`) to correct Whisper's drift on fast speech.
- **Fallback**: If Whisper returns no words for a sentence, timing is distributed proportionally by character length (`distributeTimingToWords`).

---

## File Processing Pipeline

1. **Client-side** (`pdf-client.ts`): Extracts text from PDFs using `pdfjs-dist`.
2. **Server-side** (`file-parser.ts`): DOCX via `mammoth`, TXT via `fs`, images via OpenAI vision model.
3. **API**: Files uploaded via `FormData` to `/api/extract-text`, temporarily saved to OS tmpdir, processed, then deleted.
4. **Supported formats**: `.txt`, `.docx`, `.pdf`, `.jpg`, `.jpeg`, `.png`.

---

## Environment Variables

Refer to `.env.example` for all available environment variables (~7 variables).

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string |
| `AUTH_SECRET` | NextAuth secret key |
| `AUTH_URL` | NextAuth base URL |
| `AUTH_GOOGLE_ID` | Google OAuth client ID |
| `AUTH_GOOGLE_SECRET` | Google OAuth client secret |
| `TTS_API_KEY` | OpenAI-compatible API key for TTS & Whisper |
| `TTS_BASE_URL` | OpenAI-compatible base URL for TTS & Whisper |
| `VISION_API_KEY` | API key for image OCR (falls back to `TTS_API_KEY`) |
| `VISION_BASE_URL` | Base URL for vision model (falls back to `TTS_BASE_URL`) |
| `VISION_MODEL` | Vision model name (default: `gpt-4.1-mini`) |
| `STRIPE_SECRET_KEY` | Stripe secret key for payment processing |
| `STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret |
| `WELCOME_CREDITS` | Free credits for new users (default: `3`) |
| `STRIPE_PLAN_A_CREDITS` | Credits in Starter plan (default: `15`) |
| `STRIPE_PLAN_A_PRICE_HKD` | Price in HKD for Starter plan (default: `15`) |
| `STRIPE_PLAN_B_CREDITS` | Credits in Best Value plan (default: `50`) |
| `STRIPE_PLAN_B_PRICE_HKD` | Price in HKD for Best Value plan (default: `45`) |

- **Never commit** `.env` or `.env.local` files.

---

## Backend & API Patterns

### 1. API Routes

API routes are in `src/app/api/`. Key endpoints:

| Route | Methods | Purpose |
| --- | --- | --- |
| `/api/auth/[...nextauth]` | GET/POST | NextAuth handler |
| `/api/tts` | POST | Generate TTS audio from text (deducts credits) |
| `/api/tts` | GET | List user's TTS generations |
| `/api/extract-text` | POST | Extract text from uploaded file (FormData) |
| `/api/generations/[id]` | GET | Get single generation details |
| `/api/generations/[id]` | PATCH | Update generation title |
| `/api/generations/[id]` | DELETE | Delete generation and its audio file |
| `/api/audio/[id]` | GET | Serve audio file (supports Range requests for seeking) |
| `/api/stripe/checkout` | POST | Create Stripe Checkout Session |
| `/api/stripe/webhook` | POST | Handle Stripe webhooks (no auth — Stripe signs requests) |
| `/api/stripe/plans` | GET | Return available credit plans |
| `/api/user/credits` | GET | Return user's credit balance |
| `/api/user/purchases` | GET | Return user's purchase history |

### 2. API Patterns

- All API routes (except `/api/auth/*` and `/api/stripe/webhook`) require authentication via `auth()` check.
- Return `NextResponse.json()` with appropriate HTTP status codes.
- Audio serving supports HTTP Range requests for efficient seeking in the audio player.

---

## Security & Safety

- **Secrets**: Do not hardcode API keys or credentials.
- **Auth Checks**: All API routes and dashboard pages require authentication. Auth checks are performed server-side via `auth()`.
- **File Serving**: Audio files are served only to the owning user (verified via `userId` match in database query).
- **Temp Files**: Uploaded files are stored in OS tmpdir and deleted after processing in a `finally` block.
- **Destructive Actions**: Avoid `rm -rf` or history rewriting in git unless explicitly requested.

---

## Agent Instructions

- **Read First**: Always read the relevant file and its neighbors before proposing edits.
- **Follow Patterns**: If adding a new component, look at existing components in `src/components/` for reference implementations.
- **Keep it Focused**: Make small, cohesive changes. Avoid unrelated refactors.
- **Validate**: Run `npm run lint` and `npm run build` to ensure your changes don't break the build.
- **Database Changes**: If modifying database schema, update `src/lib/db/schema.ts` and run `npm run db:generate` to create a migration. Also add a SQL migration in `scripts/` following the existing naming convention.
- **API Routes**: New API routes should follow existing patterns — use `auth()` for authentication, Drizzle ORM for database access, and proper error handling with appropriate HTTP status codes.
- **Communication**: Summarize what changed, where, and why. Call out tradeoffs, assumptions, and known limitations. If validation could not be run, say so explicitly.
- **Clarity**: Prefer clarity and simplicity over cleverness. Preserve existing behavior unless the task explicitly requires changes.
- **UI Consistency**: Ensure all new UI elements support both light and dark modes using Tailwind `dark:` classes.

## This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
