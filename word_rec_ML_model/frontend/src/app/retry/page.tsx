"use client";
import { useRouter } from "next/navigation";

export default function RetryPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center gap-8 px-4">
      <div className="flex flex-col items-center gap-4 text-center max-w-sm">
        {/* アイコン */}
        <div className="w-20 h-20 rounded-full bg-gray-800 flex items-center justify-center text-4xl">
          🎙️
        </div>

        <h1 className="text-2xl font-bold text-white">音が検出できませんでした</h1>
        <p className="text-gray-400 text-sm leading-relaxed">
          録音中の音量が小さすぎて、単語を認識できませんでした。<br />
          マイクに近づいて、もう一度はっきりと発音してください。
        </p>
      </div>

      <button
        onClick={() => router.push("/")}
        className="px-8 py-3 rounded-full bg-indigo-600 hover:bg-indigo-500
                   text-white font-semibold transition-colors shadow-lg"
      >
        もう一度録音する
      </button>
    </div>
  );
}
