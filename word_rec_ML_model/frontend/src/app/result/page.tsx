"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { InferenceResult } from "@/types";

export default function ResultPage() {
  const router = useRouter();

  // useState の遅延初期化: 初回レンダリング時に一度だけ実行される
  // useEffect + setState の組み合わせを避けることで警告を解消
  const [result] = useState<InferenceResult | null>(() => {
    const raw = sessionStorage.getItem("inferenceResult");
    if (!raw) return null;
    sessionStorage.removeItem("inferenceResult");
    return JSON.parse(raw) as InferenceResult;
  });

  // リダイレクトは副作用なので useEffect に残す
  useEffect(() => {
    if (!result) {
      router.replace("/");
    }
  }, [result, router]);

  if (!result) return <p>読み込み中...</p>;

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center gap-8 px-4">
      <header>
        <h1 className="text-3xl font-bold text-indigo-400">推論結果</h1>
      </header>

      <section className="bg-gray-800 rounded-2xl p-8 w-full max-w-md shadow-lg">
        <p className="text-2xl font-bold text-center text-white mb-6">
          {result.message}
        </p>
        <p className="text-gray-400 text-sm mb-3">他の候補:</p>
        <ul className="flex flex-col gap-2">
          {result.other_candidates.map((c) => (
            <li
              key={c.label}
              className="flex justify-between bg-gray-700 rounded-lg px-4 py-2 text-sm"
            >
              <span>{c.label}</span>
              <span className="text-indigo-300">{c.prob}</span>
            </li>
          ))}
        </ul>
      </section>

      <button
        onClick={() => router.push("/")}
        className="px-6 py-3 rounded-full border border-indigo-500 text-indigo-400
                   hover:bg-indigo-500 hover:text-white transition-colors"
      >
        もう一度録音する
      </button>
    </div>
  );
}
