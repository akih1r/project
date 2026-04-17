import { InferenceResult as InferenceResultType } from "@/types";

type InferenceResultProps = {
  result: InferenceResultType | null;
  isLoading: boolean;
};

export function InferenceResult({ result, isLoading }: InferenceResultProps) {
    if (isLoading) {
        return (
            <section>
                <p>推論中です。しばらくお待ちください...</p>
            </section>
        );
    }

    if (!result) return null;

    return (
        <section>
            <p>{result.message}</p>
            <p>他の候補:</p>
            <ul>
                {result.other_candidates.map((c) => (
                <li key={c.label}>
                    {c.label}: {c.prob}
                </li>
                ))}
            </ul>
        </section>
    )
}