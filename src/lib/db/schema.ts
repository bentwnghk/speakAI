import {
  index,
  integer,
  jsonb,
  pgEnum,
  pgTable,
  real,
  text,
  timestamp,
  varchar,
} from "drizzle-orm/pg-core";

export const users = pgTable("user", {
  id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  name: text("name"),
  email: text("email").notNull().unique(),
  emailVerified: timestamp("emailVerified", { mode: "date" }),
  image: text("image"),
});

export const accounts = pgTable(
  "account",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    userId: text("userId")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    type: text("type").notNull(),
    provider: text("provider").notNull(),
    providerAccountId: text("providerAccountId").notNull(),
    refresh_token: text("refresh_token"),
    access_token: text("access_token"),
    expires_at: integer("expires_at"),
    token_type: text("token_type"),
    scope: text("scope"),
    id_token: text("id_token"),
    session_state: text("session_state"),
  },
  (table) => [
    index("account_provider_idx").on(table.provider, table.providerAccountId),
  ]
);

export const sessions = pgTable("session", {
  id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  sessionToken: text("sessionToken").notNull().unique(),
  userId: text("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  expires: timestamp("expires", { mode: "date" }).notNull(),
});

export const verificationTokens = pgTable(
  "verification_token",
  {
    identifier: text("identifier").notNull(),
    token: text("token").notNull().unique(),
    expires: timestamp("expires", { mode: "date" }).notNull(),
  },
  (table) => [
    index("verification_token_idx").on(table.identifier, table.token),
  ]
);

export const voiceEnum = pgEnum("voice", [
  "nova",
  "alloy",
  "phoebe",
  "adam",
  "ava",
  "ollie",
]);

export const generations = pgTable("generation", {
  id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  userId: text("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  title: varchar("title", { length: 500 }).notNull(),
  transcript: text("transcript").notNull(),
  voice: voiceEnum("voice").notNull().default("nova"),
  speed: integer("speed").notNull().default(100),
  audioPath: text("audioPath").notNull(),
  segments: text("segments"),
  ttsCost: text("ttsCost"),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
  expiresAt: timestamp("expiresAt", { mode: "date" }),
});

export const credits = pgTable("credits", {
  userId: text("userId")
    .primaryKey()
    .references(() => users.id, { onDelete: "cascade" }),
  balance: real("balance").notNull().default(0),
  updatedAt: timestamp("updatedAt", { mode: "date" }).defaultNow().notNull(),
});

export const creditTransactions = pgTable("credit_transactions", {
  id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  userId: text("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  amount: real("amount").notNull(),
  type: varchar("type", { length: 30 }).notNull(),
  description: text("description"),
  stripeSessionId: text("stripeSessionId"),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
});

export const userSettings = pgTable("user_settings", {
  userId: text("userId")
    .primaryKey()
    .references(() => users.id, { onDelete: "cascade" }),
  theme: varchar("theme", { length: 10 }).notNull().default("system"),
  locale: varchar("locale", { length: 10 }).notNull().default("en"),
  updatedAt: timestamp("updatedAt", { mode: "date" }).defaultNow().notNull(),
});

export const signInLogs = pgTable("sign_in_logs", {
  id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  userId: text("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  provider: text("provider").notNull().default("google"),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
});

export const assessments = pgTable(
  "assessments",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    userId: text("userId")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    referenceText: text("referenceText").notNull(),
    recognizedText: text("recognizedText").notNull(),
    durationMs: integer("durationMs").notNull(),
    accuracyScore: real("accuracyScore").notNull(),
    fluencyScore: real("fluencyScore").notNull(),
    completenessScore: real("completenessScore").notNull(),
    prosodyScore: real("prosodyScore"),
    pronScore: real("pronScore").notNull(),
    words: jsonb("words").notNull(),
    phonemes: jsonb("phonemes"),
    syllables: jsonb("syllables"),
    audioPath: text("audioPath"),
    referenceAudioPath: text("referenceAudioPath"),
    cost: real("cost").notNull(),
    createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
    expiresAt: timestamp("expiresAt", { mode: "date" }),
  },
  (table) => [index("assessments_user_idx").on(table.userId)]
);

export const purchases = pgTable("purchases", {
  id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
  userId: text("userId")
    .notNull()
    .references(() => users.id, { onDelete: "cascade" }),
  stripeSessionId: text("stripeSessionId").notNull().unique(),
  stripePaymentIntentId: text("stripePaymentIntentId"),
  planName: varchar("planName", { length: 20 }).notNull(),
  creditsAmount: real("creditsAmount").notNull(),
  amountHKD: real("amountHKD").notNull(),
  status: varchar("status", { length: 20 }).notNull().default("pending"),
  createdAt: timestamp("createdAt", { mode: "date" }).defaultNow().notNull(),
});
