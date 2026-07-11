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
│   │   ├── admin/              # Admin dashboard (admin-only, server-side guard in layout)
│   │   │   ├── layout.tsx      # Admin auth guard (checks ADMIN_EMAIL)
│   │   │   └── page.tsx        # Admin page with TTS + Assessment tabs
│   │   ├── assessment/         # Pronunciation assessment page
│   │   ├── credits/            # Credit purchase page with Stripe
│   │   └── history/            # Generation + assessment history (tabbed)
│   └── api/                    # API route handlers (see Backend section)
├── components/
│   ├── assessment/             # Pronunciation assessment components
│   │   ├── assessment-form.tsx      # Main assessment form (recording, scoring, transcript)
│   │   ├── assessment-history.tsx   # Paginated assessment history list with detail view
│   │   ├── error-summary.tsx        # Filterable error-type and accuracy-tier chips
│   │   ├── play-word-button.tsx     # Inline word pronunciation via /api/tts/word
│   │   ├── recording-controls.tsx   # Mic/stop button with animated pulse, elapsed timer
│   │   ├── reference-audio-section.tsx # Generate/listen to TTS of reference text
│   │   ├── score-overview.tsx       # SVG circular gauge + 4 score bars with InfoTip
│   │   ├── transcript-view.tsx      # Interactive word-by-word color-coded transcript
│   │   └── word-detail.tsx          # Word/phoneme/syllable breakdown with IPA, confused phonemes
│   ├── auth-provider.tsx       # Client provider: SessionProvider + CreditsProvider wrapper
│   ├── audio-player.tsx        # Audio playback with karaoke-style highlighting
│   ├── dashboard-page-header.tsx # Extracted header for dashboard page
│   ├── file-upload.tsx         # File upload (PDF, DOCX, TXT, images)
│   ├── header.tsx              # App header with navigation (Home, History, Assessment, Admin) + credits
│   ├── history-list.tsx        # TTS generation history list with pagination, expiry badges
│   ├── history-page-header.tsx # Extracted header for history page
│   ├── history-tabs.tsx        # Tabbed switcher for Audio / Assessment history
│   ├── karaoke-subtitle.tsx    # Animated wave subtitle on dashboard page (motion)
│   ├── karaoke-text.tsx        # Word-by-word karaoke text display
│   ├── selection-popover.tsx   # Floating "Practice reading aloud" button on text selection
│   ├── speed-slider.tsx        # Speed control slider
│   ├── tts-form.tsx            # Main TTS form (text input, voice, speed, file upload, selection popover)
│   ├── voice-select.tsx        # Voice selection dropdown
│   ├── landing/                # Public landing/marketing page
│   │   └── landing-page.tsx    # Landing page with TTS + Assessment feature showcase
│   └── ui/                     # Shadcn UI primitives (do not modify directly)
├── lib/
│   ├── admin.ts                # getAdminEmails(), isAdminEmail() from ADMIN_EMAIL env var
│   ├── auth.ts                 # NextAuth v5 config (Google OAuth, JWT strategy, Drizzle adapter, isAdmin)
│   ├── constants.ts            # Voice mappings, supported file extensions
│   ├── utils.ts                # cn() utility (clsx + tailwind-merge)
│   ├── pdf-client.ts           # Client-side PDF processing (pdfjs-dist)
│   ├── file-parser.ts          # Server-side file text extraction (DOCX, TXT, images)
│   ├── tts.ts                  # TTS generation + Azure Speech word-boundary alignment + round-robin endpoint pool
│   ├── word-tts.ts             # Single-word TTS synthesizer (separate round-robin, slower rate for clarity)
│   ├── stripe.ts               # Stripe client singleton + credit plan definitions
│   ├── i18n/
│   │   ├── index.ts            # getDictionary(), Locale type, TranslationKeys type
│   │   └── locales/
│   │       ├── en.ts           # English translations (canonical, defines TranslationKeys shape)
│   │       └── zh-TW.ts        # Traditional Chinese translations
│   └── db/
│       ├── index.ts            # Drizzle ORM setup (postgres-js driver, schema export)
│       ├── schema.ts           # 11 tables (see Database section)
│       └── credits.ts          # Credit engine: grant, deduct, refund, purchase operations
├── sw/
│   └── index.ts                # Serwist service worker for PWA offline support
├── hooks/
│   ├── use-credits.tsx         # React context for credit balance (app-wide)
│   ├── use-settings.tsx        # React context for theme + locale (UserSettingsProvider)
│   └── use-text-selection.tsx  # Text selection tracking hook for textarea (mirror div positioning)
└── types/
    ├── assessment.ts           # Pronunciation assessment types (scores, words, phonemes, filters, helpers)
    ├── karaoke.ts              # WordTimestamp, Segment types for karaoke display
    ├── next-auth.d.ts          # Session/JWT type augmentation (isAdmin)
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
- **Provider Hierarchy**: `AuthProvider` wraps `SessionProvider` from `next-auth/react` and `CreditsProvider`. `UserSettingsProvider` wraps theme/locale state (see Theme section).

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
- **Schema**: All tables defined in `src/lib/db/schema.ts` (11 tables: `user`, `account`, `session`, `verification_token`, `generation`, `credits`, `credit_transactions`, `purchases`, `user_settings`, `sign_in_logs`, `assessments`). The `generation` table stores TTS generation history with voice (as a Postgres enum), speed, audio path, karaoke segments (JSON), cost, and `expiresAt`. The `assessments` table stores pronunciation assessment results with reference/recognized text, 5-dimension scores (Accuracy, Fluency, Completeness, Prosody, PronScore), words/phonemes (JSONB), recording audio path, reference audio path, cost, and `expiresAt`. The `credits` table stores per-user balance (1:1 with users). `credit_transactions` is an append-only ledger. `purchases` tracks Stripe payment lifecycle. `user_settings` stores per-user theme and locale preferences. `sign_in_logs` records user sign-in events.
- **Migrations**: SQL migration files in `scripts/` (e.g., `init-db.sql`, `add-segments-column.sql`). Drizzle migrations in `drizzle/`.
- **Config**: `drizzle.config.ts` at project root.

---

## TTS & Audio Pipeline

### Text-to-Speech

- **`src/lib/tts.ts`**: Core TTS logic using **Azure Speech SDK** (`microsoft-cognitiveservices-speech-sdk`). Synthesizes MP3 audio and collects word-boundary events from the synthesis engine for karaoke timing.
- **Voices**: 6 voices available — Nova (Female 1), Alloy (Male 1), Phoebe (Female 2), Adam (Male 2), Ava (Female 3), Ollie (Male 3). Mapped via `AZURE_VOICE_MAP` in `src/lib/tts.ts`. Individual voice names are overridable via env vars (e.g. `AZURE_SPEECH_VOICE_FEMALE_1`).
- **Text Chunking**: Long text is split into chunks (max 4000 chars) respecting paragraph and sentence boundaries. Chunks are processed **sequentially** so cumulative audio offsets are deterministic.
- **Audio Output**: Generated MP3 files stored in `data/audio/` with nanoid-based filenames.

### Endpoint Rotation (Load Balancing)

- **Multi-endpoint pool**: Supports an arbitrary number of Azure Speech endpoints via numbered env vars (`AZURE_SPEECH_KEY_1` + `AZURE_SPEECH_REGION_1` through `_N`). The unnumbered `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` pair is a single-endpoint fallback for backward compatibility.
- **Round-robin**: A module-level cursor (`rrCursor`) is incremented for each synthesized chunk. Chunks within a long text and across consecutive requests are dispatched to successive endpoints in the pool, evenly distributing API character costs and call volume.
- **Configuration**: `loadAzureEndpoints()` runs once at module load, scanning `_1` through `_20`.

### Audio Alignment (Karaoke)

- **Azure Word Boundaries**: During synthesis, the Azure Speech SDK emits `wordBoundary` events with per-word start/end timestamps. These are ground-truth timing from the same engine that produces the audio.
- **Segment Building**: `buildSegmentsFromAzureBoundaries()` maps boundary events onto source text sentences/words. Per-word timing is derived via character-position proportional alignment (`mapWordsToTimings`) to handle minor vocabulary differences between source and synthesis output.
- **Fallback**: If Azure returns fewer boundary events than source words for a sentence, timing is distributed proportionally by character length (`distributeTimingToWords`).

---

## Pronunciation Assessment

A full pronunciation coaching system built on Azure Speech SDK's Pronunciation Assessment API. Users read reference text aloud, receive scored feedback across 5 dimensions, and can drill into word/phoneme-level detail.

### Types (`src/types/assessment.ts`)

- **ErrorType**: `"None" | "Omission" | "Insertion" | "Mispronunciation" | "UnexpectedBreak" | "MissingBreak" | "Monotone"`
- **PronunciationScores**: 5-dimension scoring (Accuracy, Fluency, Completeness, Prosody, PronScore)
- **AccuracyTier**: `"Excellent" | "Good" | "Fair"` — derived from score thresholds (>=90, >=80, rest)
- **AssessmentFilter**: Union type supporting `"All"`, error types, and accuracy tiers like `"None:Excellent"`
- **GranularityLevel**: `"FullText" | "Word" | "Phoneme"`
- **RecordingMode**: `"auto" | "manual"` — auto-detect end vs. user-controlled stop
- **RecordingState**: `"idle" | "recording" | "processing" | "done"`
- Helper functions: `getAccuracyTier()`, `filterWords()`

### Recording Modes

- **Auto mode**: Single-shot `recognizeOnceAsync` — Azure detects end of speech automatically.
- **Manual mode**: Continuous `recognizeContinuousAsync` with user-controlled stop. Combines multiple recognition results, averages scores, applies client-side miscue correction for unmatched words (marks as Insertion).

### Cost Model

- Assessment cost = `duration_hours * ASSESSMENT_PRICE_USD_PER_HOUR * 7.8 HKD`. Configurable via `ASSESSMENT_PRICE_USD_PER_HOUR` env var (default `1.00`).
- Additional costs for word pronunciation playback (`/api/tts/word`) and reference audio generation are accumulated incrementally via PATCH on the assessment record.
- Recordings saved to `data/recording/` with nanoid filenames. Reference audio saved to standard TTS output path.

### Word-Level TTS (`src/lib/word-tts.ts`)

Dedicated single-word TTS synthesizer with its own Azure endpoint pool (separate round-robin cursor from main TTS). Synthesizes via SSML with 15% slower rate for clearer pronunciation. XML-escapes the word. Used by `PlayWordButton` and `TranscriptView` for inline pronunciation playback.

### Text Selection → Assessment Flow

- **`useTextSelectionPopover`** (`src/hooks/use-text-selection.tsx`): Tracks text selection within a `<textarea>`. Uses a hidden "mirror" `<div>` to compute selection bounding rect. Debounces `selectionchange` events (150ms). Auto-adjusts popup position to avoid screen edges.
- **`SelectionPopover`** (`src/components/selection-popover.tsx`): Floating button that appears above selected text, navigating to `/assessment?text=<selected_text>`.

---

## Admin System

### Architecture

- **`src/lib/admin.ts`**: `getAdminEmails()` parses comma-separated `ADMIN_EMAIL` env var. `isAdminEmail()` checks if email is in admin list.
- **Session augmentation** (`src/types/next-auth.d.ts`): `isAdmin` boolean added to `Session.user` and `JWT` interfaces. Set in `auth.ts` JWT callback via `isAdminEmail()`.
- **Admin layout** (`src/app/(dashboard)/admin/layout.tsx`): Server-side guard — calls `auth()` + `isAdminEmail()`, redirects to `/` if unauthorized.

### Admin Access Rules

- All admin API routes (`/api/admin/*`) check `isAdminEmail()` and return 403 if not admin.
- All user-facing API routes (assessment GET, audio serving) grant admin users access to any record regardless of ownership or expiry.

---

## Audio Retention & Data Expiry

### Expiry System

- **TTS audio**: `generations.expiresAt` set on creation based on `AUDIO_RETENTION_DAYS` env var (default 365).
- **Assessment recordings**: `assessments.expiresAt` set only when `audioPath` is present, based on `RECORDING_RETENTION_DAYS` (falls back to `AUDIO_RETENTION_DAYS`).
- All user-facing GET endpoints exclude expired records via `or(isNull(expiresAt), gt(expiresAt, new Date()))`.

### Cleanup API (`/api/cleanup`)

- POST endpoint deletes expired generations and assessments (and their audio files) in a single pass.
- Returns `{ deletedGenerations, deletedAssessments }`.
- No auth required if `CLEANUP_SECRET` env var is unset; otherwise requires `Bearer` token.

### Expiry UI

- History lists show expiry badges ("Xd left") with red styling when <=14 days remaining. Near-expiry cards get red border.

---

## History Page (Tabbed Interface)

The history page uses a tabbed interface with "Audio" and "Assessment" tabs:

- **`history-tabs.tsx`**: Reads `?tab=assessment` search param for default tab.
- **`history-list.tsx`**: TTS generation history with pagination (page size selector: 10/20/30/50), expiry badges, and text selection popover in detail view.
- **`assessment-history.tsx`**: Paginated assessment history with detail drill-down, audio playback, and full score/transcript display.
- History page uses `Suspense` boundary because `HistoryTabs` uses `useSearchParams()`.

---

## File Processing Pipeline

1. **Client-side** (`pdf-client.ts`): Extracts text from PDFs using `pdfjs-dist`.
2. **Server-side** (`file-parser.ts`): DOCX via `mammoth`, TXT via `fs`, images via OpenAI vision model.
3. **API**: Files uploaded via `FormData` to `/api/extract-text`, temporarily saved to OS tmpdir, processed, then deleted.
4. **Supported formats**: `.txt`, `.docx`, `.pdf`, `.jpg`, `.jpeg`, `.png`.

---

## Internationalization (i18n)

The project uses a **custom-built, lightweight i18n system** with no external library. All translations are fully type-safe via TypeScript.

### Architecture

- **Core module**: `src/lib/i18n/index.ts` — exports `getDictionary(locale)`, `Locale` type (`"en" | "zh-TW"`), `TranslationKeys` type.
- **Dictionaries**: TypeScript `const` objects in `src/lib/i18n/locales/`:
  - `en.ts` — English (canonical/source; defines `TranslationKeys` shape)
  - `zh-TW.ts` — Traditional Chinese (繁體中文; satisfies `TranslationKeys`)
- **Type safety**: All locale files must satisfy the `TranslationKeys` type derived from `en.ts`. Adding a key to `en.ts` will cause a type error in `zh-TW.ts` until the translation is added.

### Supported Locales

| Code | Language |
| --- | --- |
| `en` | English (default) |
| `zh-TW` | Traditional Chinese (繁體中文) |

### Translation Namespaces

Dictionaries are organized into namespaces: `common`, `header`, `dashboard`, `tts`, `history`, `credits`, `assessment`, `admin`, `landing`, `install`, `settings`.

### Usage Patterns

- **Client components**: Call `useUserSettings()` from `@/hooks/use-settings` to get `t` (the translation dictionary). Access translations via dot notation: `t.common.appName`, `t.tts.generate`.
- **Server components**: Call `getDictionary(locale)` directly from `@/lib/i18n`.
- **Parameterized strings**: Use JavaScript `.replace()`. Two conventions exist in the codebase — `${var}` and `{var}`. Example: `t.tts.cost.replace("${cost}", data.ttsCost)`.

### Locale Resolution

1. **Default**: `"en"` (React state initial value).
2. **localStorage**: Reads key `"speakai-locale"` on mount — takes priority.
3. **Database sync**: On auth, fetches from `GET /api/user/settings` (`user_settings.locale` column). Applied only if no localStorage preference exists.
4. **User change**: Via Settings Dialog → updates React state, localStorage, `document.documentElement.lang` (`"zh-Hant"` for `zh-TW`), and persists via `PUT /api/user/settings`.
5. **No URL-based routing**: No middleware, no path prefixes like `/en/...`. Locale is entirely user-preference-driven.

### Adding a New Locale

1. Create `src/lib/i18n/locales/<locale>.ts` importing `TranslationKeys` from `en.ts` and using `satisfies TranslationKeys`.
2. Add the locale to the `Locale` type union and `dictionaries` map in `src/lib/i18n/index.ts`.
3. Add a Zod enum entry in `src/app/api/user/settings/route.ts`.
4. Add translated labels in the `settings` namespace of all locale files (e.g., `settings.langNewLocale`).

---

## Theme / Dark Mode

The project uses a **custom dark/light mode system** (no `next-themes`). Theme is stored in three layers: localStorage (primary), PostgreSQL (cross-device), and an inline `<script>` for FOUC prevention.

### Architecture

- **Theme type**: `"system" | "light" | "dark"` (defined in `src/hooks/use-settings.tsx`).
- **Application**: The `.dark` CSS class is toggled on `document.documentElement` (`<html>`).
- **CSS variables**: OKLCH color space, defined in `src/app/globals.css`:
  - Light mode on `:root`
  - Dark mode on `.dark`
  - Semantic tokens: `--background`, `--foreground`, `--card`, `--primary`, `--secondary`, `--muted`, `--accent`, `--destructive`, `--border`, `--input`, `--ring`, etc.
- **Tailwind v4 integration**: `@custom-variant dark (&:where(.dark, .dark *))` in `globals.css` — no `tailwind.config.ts`.

### FOUC Prevention

An inline `<script>` in `src/app/layout.tsx` `<head>` runs synchronously before paint. It reads `localStorage('speakai-theme')`, resolves system preference via `matchMedia('(prefers-color-scheme: dark)')`, and adds/removes the `.dark` class. The `<html>` tag has `suppressHydrationWarning` for this reason.

### Theme Resolution Flow

1. **Page load** → inline `<script>` reads localStorage + OS preference → applies `.dark` class instantly.
2. **React hydration** → `UserSettingsProvider` reads localStorage, updates React state. If authenticated and no local preference, fetches server settings via `GET /api/user/settings`.
3. **User switches theme** → `setTheme()` in `use-settings.tsx`:
   - Updates React state
   - Writes to `localStorage('speakai-theme')`
   - Calls `applyTheme()` to toggle `.dark` class on `<html>`
   - Fires `PUT /api/user/settings` to persist to database
4. **System theme live updates** (when in "system" mode) → `matchMedia` change listener re-applies theme in real-time.

### Persistence Layers

| Layer | Key/Column | Priority |
| --- | --- | --- |
| localStorage | `"speakai-theme"` | Highest (instant, always checked first) |
| PostgreSQL | `user_settings.theme` | Secondary (synced on login if no local pref) |
| Inline `<script>` | Reads localStorage + OS pref | FOUC prevention only |

### Settings UI

- **Settings Dialog** (`src/components/settings-dialog.tsx`): Shadcn `<Select>` dropdown with System/Light/Dark options.
- **Trigger**: Settings menu item in the header user dropdown (`src/components/header.tsx`).

### Provider Hierarchy

```
<AuthProvider>              (next-auth SessionProvider wrapper)
  <UserSettingsProvider>    (provides theme, locale, t, setLocale, setTheme)
    {children}
    <Toaster />
    <InstallPrompt />
  </UserSettingsProvider>
</AuthProvider>
```

### Conventions

- Use Tailwind `dark:` classes for all new UI elements. Use the semantic CSS variables (e.g., `bg-background`, `text-foreground`, `border-border`) which automatically adapt to the active theme.
- Avoid hardcoding colors — use the OKLCH CSS variables or Tailwind's semantic color utilities.
- Test all UI in both light and dark modes.

---

## User Settings

User preferences (theme and locale) are persisted in the `user_settings` table:

```sql
CREATE TABLE user_settings (
  userId TEXT NOT NULL PRIMARY KEY REFERENCES user(id) ON DELETE CASCADE,
  theme VARCHAR(10) NOT NULL DEFAULT 'system',
  locale VARCHAR(10) NOT NULL DEFAULT 'en',
  updatedAt TIMESTAMP NOT NULL DEFAULT NOW()
);
```

- **API**: `GET/PUT /api/user/settings` (`src/app/api/user/settings/route.ts`).
- **Hook**: `useUserSettings()` from `@/hooks/use-settings` provides `theme`, `setTheme`, `locale`, `setLocale`, `t`, `loading`.
- **Context**: `UserSettingsProvider` wraps the app inside `AuthProvider`.

---

## Environment Variables

Refer to `.env.example` for all available environment variables.

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string |
| `AUTH_SECRET` | NextAuth secret key |
| `AUTH_URL` | NextAuth base URL |
| `AUTH_GOOGLE_ID` | Google OAuth client ID |
| `AUTH_GOOGLE_SECRET` | Google OAuth client secret |
| `AZURE_SPEECH_KEY_1`…`_N` | Azure Speech subscription keys (round-robin pool) |
| `AZURE_SPEECH_REGION_1`…`_N` | Azure Speech regions (must match corresponding `_KEY`) |
| `AZURE_SPEECH_KEY` | Single-endpoint fallback (when no numbered vars set) |
| `AZURE_SPEECH_REGION` | Single-endpoint fallback region |
| `AZURE_SPEECH_PRICE_USD_PER_1M_CHARS` | Cost per 1M chars in USD (default: `16`) |
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
| `ADMIN_EMAIL` | Comma-separated admin email addresses |
| `ASSESSMENT_PRICE_USD_PER_HOUR` | Cost per hour of assessment audio in USD (default: `1.00`) |
| `AUDIO_RETENTION_DAYS` | Days before TTS audio is auto-deleted (default: `365`) |
| `RECORDING_RETENTION_DAYS` | Days before assessment recordings are deleted (falls back to `AUDIO_RETENTION_DAYS`) |
| `CLEANUP_SECRET` | Optional Bearer token for `/api/cleanup` endpoint (no auth if unset) |

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
| `/api/tts/word` | POST | Synthesize single word for pronunciation playback (returns audio + credit headers) |
| `/api/extract-text` | POST | Extract text from uploaded file (FormData) |
| `/api/generations/[id]` | GET | Get single generation details |
| `/api/generations/[id]` | PATCH | Update generation title |
| `/api/generations/[id]` | DELETE | Delete generation and its audio file |
| `/api/audio/[id]` | GET | Serve audio file (supports Range requests for seeking) |
| `/api/assessment` | POST | Save pronunciation assessment (FormData: JSON + audio blob, deducts credits) |
| `/api/assessment` | GET | List user's assessments (paginated, excludes expired) |
| `/api/assessment/[id]` | GET | Get full assessment detail (admin bypasses ownership/expiry checks) |
| `/api/assessment/[id]` | DELETE | Delete assessment and its audio files |
| `/api/assessment/[id]` | PATCH | Update reference audio path or increment cost |
| `/api/assessment/[id]/audio` | GET | Serve recording audio (mp4/m4a/webm, Range requests, admin bypass) |
| `/api/assessment/[id]/reference-audio` | GET | Serve reference audio MP3 (admin bypass) |
| `/api/speech/token` | GET | Issue Azure Speech auth token (round-robin, for client-side assessment) |
| `/api/admin/assessments` | GET | Paginated list of all assessments with user info (admin-only) |
| `/api/admin/generations` | GET | Admin-only generation listing |
| `/api/admin/purchases` | GET | Admin-only purchase listing |
| `/api/admin/sign-ins` | GET | Admin-only sign-in log listing |
| `/api/stripe/checkout` | POST | Create Stripe Checkout Session |
| `/api/stripe/webhook` | POST | Handle Stripe webhooks (no auth — Stripe signs requests) |
| `/api/stripe/plans` | GET | Return available credit plans |
| `/api/cleanup` | POST | Delete expired generations + assessments and their audio files |
| `/api/user/credits` | GET | Return user's credit balance |
| `/api/user/purchases` | GET | Return user's purchase history |
| `/api/user/settings` | GET | Return user's theme and locale preferences |
| `/api/user/settings` | PUT | Update user's theme and/or locale preferences |

### 2. API Patterns

- All API routes (except `/api/auth/*` and `/api/stripe/webhook`) require authentication via `auth()` check.
- Admin routes (`/api/admin/*`) additionally check `isAdminEmail(session.user.email)` and return 403 if unauthorized.
- User-facing routes (assessment, audio serving) grant admin users access to any record regardless of ownership or expiry.
- Return `NextResponse.json()` with appropriate HTTP status codes.
- Audio serving supports HTTP Range requests for efficient seeking in the audio player.

---

## Security & Safety

- **Secrets**: Do not hardcode API keys or credentials.
- **Auth Checks**: All API routes and dashboard pages require authentication. Auth checks are performed server-side via `auth()`.
- **Admin Checks**: Admin routes (`/api/admin/*`, `/admin/*`) additionally verify `isAdminEmail()`.
- **File Serving**: Audio files are served only to the owning user (verified via `userId` match in database query). Admin users bypass ownership/expiry checks.
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
