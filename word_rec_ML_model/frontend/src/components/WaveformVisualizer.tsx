"use client";
import React, { useEffect, useRef } from "react";

type Props = {
  analyserRef: React.RefObject<AnalyserNode | null>;
  isRecording: boolean;
  remainingTime: number;
  recordDuration: number;
};

export function WaveformVisualizer({ analyserRef, isRecording, remainingTime, recordDuration }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  // 各フレームで取得した振幅（RMS）を蓄積するリスト
  const waveHistoryRef = useRef<number[]>([]);
  // requestAnimationFrame のタイムスタンプ基準で録音開始時刻を記録
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // isRecordingが変わるたびに履歴リセット
    waveHistoryRef.current = [];
    startTimeRef.current = null;

    const draw = (timestamp: number) => {
      animFrameRef.current = requestAnimationFrame(draw);
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      if (isRecording && analyserRef.current) {
        // 録音開始時刻を初回フレームで記録
        if (startTimeRef.current === null) startTimeRef.current = timestamp;
        const elapsed = Math.min((timestamp - startTimeRef.current) / 1000, recordDuration);
        const playheadX = (elapsed / recordDuration) * width;

        // 現在フレームのRMS振幅（0〜1）を計算して蓄積
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          const norm = (dataArray[i] - 128) / 128; // -1〜1 に正規化
          sum += norm * norm;
        }
        const amplitude = Math.sqrt(sum / bufferLength);
        waveHistoryRef.current.push(amplitude);

        // playhead左側: 蓄積した波形を縦バーとして描画
        const sampleCount = waveHistoryRef.current.length;
        ctx.strokeStyle = "#818cf8"; // indigo-400
        ctx.lineWidth = 1.5;
        for (let i = 0; i < sampleCount; i++) {
          const x = (i / sampleCount) * playheadX;
          const amp = waveHistoryRef.current[i];
          const barHeight = Math.max(amp * height * 3, 2); // 最小2pxで視認性確保
          ctx.beginPath();
          ctx.moveTo(x, (height - barHeight) / 2);
          ctx.lineTo(x, (height + barHeight) / 2);
          ctx.stroke();
        }

        // playhead右側: グレーの水平線
        ctx.beginPath();
        ctx.strokeStyle = "#374151"; // gray-700
        ctx.lineWidth = 1;
        ctx.moveTo(playheadX, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // playhead: 白い縦棒
        ctx.beginPath();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.moveTo(playheadX, 4);
        ctx.lineTo(playheadX, height - 4);
        ctx.stroke();
      } else {
        // 待機中: グレーの水平線のみ
        ctx.beginPath();
        ctx.strokeStyle = "#4b5563"; // gray-600
        ctx.lineWidth = 2;
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
      }
    };

    animFrameRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [analyserRef, isRecording, recordDuration]);

  return (
    <div className="flex flex-col items-center gap-2 w-full max-w-md">
      <canvas
        ref={canvasRef}
        width={400}
        height={100}
        className="w-full rounded-xl bg-gray-800"
      />
      <p className="text-gray-400 text-xs">
        {isRecording ? `残り ${remainingTime}秒` : `録音時間: ${recordDuration}秒`}
      </p>
    </div>
  );
}
