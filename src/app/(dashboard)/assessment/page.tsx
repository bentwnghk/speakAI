import type { Metadata } from "next";
import { AssessmentForm } from "@/components/assessment/assessment-form";

export const metadata: Metadata = {
  title: "Speaking Assessment - Mr.\u{1F196} SpeakAI",
};

const pricePerHourUsd = parseFloat(process.env.ASSESSMENT_PRICE_USD_PER_HOUR || "1.00");
const pricePerMinHkd = Math.round((pricePerHourUsd * 7.8 / 60) * 100) / 100;

export default function AssessmentPage() {
  return <AssessmentForm pricePerMinHkd={pricePerMinHkd} />;
}
