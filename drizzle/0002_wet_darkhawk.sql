CREATE TABLE "assessments" (
	"id" text PRIMARY KEY NOT NULL,
	"userId" text NOT NULL,
	"referenceText" text NOT NULL,
	"recognizedText" text NOT NULL,
	"durationMs" integer NOT NULL,
	"accuracyScore" real NOT NULL,
	"fluencyScore" real NOT NULL,
	"completenessScore" real NOT NULL,
	"prosodyScore" real,
	"pronScore" real NOT NULL,
	"words" jsonb NOT NULL,
	"phonemes" jsonb,
	"audioPath" text,
	"referenceAudioPath" text,
	"referenceSegments" jsonb,
	"feedback" jsonb,
	"stress" jsonb,
	"cost" real NOT NULL,
	"createdAt" timestamp DEFAULT now() NOT NULL,
	"expiresAt" timestamp
);
--> statement-breakpoint
CREATE TABLE "sign_in_logs" (
	"id" text PRIMARY KEY NOT NULL,
	"userId" text NOT NULL,
	"provider" text DEFAULT 'google' NOT NULL,
	"createdAt" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "assessments" ADD CONSTRAINT "assessments_userId_user_id_fk" FOREIGN KEY ("userId") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "sign_in_logs" ADD CONSTRAINT "sign_in_logs_userId_user_id_fk" FOREIGN KEY ("userId") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "assessments_user_idx" ON "assessments" USING btree ("userId");