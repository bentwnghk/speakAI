import type { Metadata } from "next";
import { AssessmentForm } from "@/components/assessment/assessment-form";

export const metadata: Metadata = {
  title: "Speaking Assessment - Mr.\u{1F196} SpeakAI",
};

const assessmentCost = parseFloat(process.env.ASSESSMENT_COST_HKD || "0.50");

export default function AssessmentPage() {
  return <AssessmentForm cost={assessmentCost} />;
}
