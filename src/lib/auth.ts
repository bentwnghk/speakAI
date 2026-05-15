import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import { DrizzleAdapter } from "@auth/drizzle-adapter";
import { db } from "./db";
import { users, signInLogs } from "./db/schema";
import { eq } from "drizzle-orm";
import { ensureCreditsRecord } from "./db/credits";
import { isAdminEmail } from "./admin";

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: {
    ...DrizzleAdapter(db),
    async createUser(data) {
      const existing = await db
        .select()
        .from(users)
        .where(eq(users.email, data.email))
        .limit(1);

      if (existing.length > 0) {
        return existing[0];
      }

      const [created] = await db.insert(users).values(data).returning();
      return created;
    },
  },
  providers: [Google],
  session: { strategy: "jwt" },
  pages: {
    signIn: "/login",
  },
  events: {
    async signIn({ user, account }) {
      if (user.id) {
        await ensureCreditsRecord(user.id);
        await db.insert(signInLogs).values({
          userId: user.id,
          provider: account?.provider || "unknown",
        });
      }
    },
  },
  callbacks: {
    session({ session, token }) {
      if (token?.sub) {
        session.user.id = token.sub;
      }
      if (session.user) {
        session.user.isAdmin = (token.isAdmin as boolean) ?? false;
      }
      return session;
    },
    jwt({ token, user }) {
      if (user) {
        token.sub = user.id;
        token.isAdmin = isAdminEmail(user.email);
      }
      return token;
    },
  },
});
