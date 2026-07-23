export {};

declare global {
  interface Window {
    __stopAssessment?: () => void;
  }
}
