export interface WordTimestamp {
  word: string;
  start: number;
  end: number;
}

export interface Segment {
  text: string;
  startTime: number;
  endTime: number;
  words: WordTimestamp[];
}
