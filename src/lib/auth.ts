import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import { DrizzleAdapter } from "@auth/drizzle-adapter";
import { db } from "./db";
import { users } from "./db/schema";
import { eq } from "drizzle-orm";

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: DrizzleAdapter(db),
  providers: [Google],
  session: { strategy: "jwt" },
  pages: {
    signIn: "/login",
  },
  callbacks: {
    async signIn({ user }) {
      if (user.email) {
        const existing = await db
          .select()
          .from(users)
          .where(eq(users.email, user.email))
          .limit(1);
        if (existing.length === 0) {
          await db.insert(users).values({
            id: user.id,
            name: user.name,
            email: user.email,
            emailVerified: "emailVerified" in user ? user.emailVerified : null,
            image: user.image,
          });
        } else {
          user.id = existing[0].id;
        }
      }
      return true;
    },
    session({ session, token }) {
      if (token?.sub) {
        session.user.id = token.sub;
      }
      return session;
    },
    jwt({ token, user }) {
      if (user) {
        token.sub = user.id;
      }
      return token;
    },
  },
});
