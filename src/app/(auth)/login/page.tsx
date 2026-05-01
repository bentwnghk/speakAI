"use client";

import { signIn } from "next-auth/react";
import { LandingPage } from "@/components/landing/landing-page";

export default function LoginPage() {
  return (
    <LandingPage onSignIn={() => void signIn("google", { callbackUrl: "/" })} />
  );
}
