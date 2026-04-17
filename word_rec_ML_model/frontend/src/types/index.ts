export type InferenceResult = {
  filename: string;
  top_result: { label: string; prob: string };
  other_candidates: { label: string; prob: string }[];
  message: string;
};