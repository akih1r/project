type RecordControlsProps = {
  isRecording: boolean;
  isLoading: boolean;
  onRecord: () => void;
};

export function RecordControls({ onRecord, isRecording, isLoading }: RecordControlsProps) {
  const disabled = isRecording || isLoading;
  return (
    <div className="flex flex-col items-center gap-4">
      <button
        onClick={onRecord}
        disabled={disabled}
        className="w-20 h-20 rounded-full bg-indigo-600 hover:bg-indigo-500
                   text-white text-3xl font-bold shadow-lg transition-colors
                   flex items-center justify-center
                   disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-indigo-600"
      >
        ▶
      </button>
      <p className="text-gray-400 text-sm">
        {isLoading ? "推論中..." : isRecording ? "録音中..." : "▶を押して録音開始"}
      </p>
    </div>
  );
}

