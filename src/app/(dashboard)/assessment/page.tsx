import type { Metadata } from "next";
import { AssessmentForm } from "@/components/assessment/assessment-form";

export const metadata: Metadata = {
  title: "Speaking Assessment - Mr.\u{1F19A} SpeakAI",
};

export default function AssessmentPage() {
  return <AssessmentForm />;
}
