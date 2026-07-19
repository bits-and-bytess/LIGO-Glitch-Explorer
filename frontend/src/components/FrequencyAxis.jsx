// A structural divider styled like a spectrogram's own frequency axis --
// thin rule, tick marks, sparse Hz labels. Used between major sections
// instead of a generic <hr>, since the content it separates is literally
// about frequency-domain analysis.
export default function FrequencyAxis({ labels = ["10", "100", "1k", "2k Hz"], className = "" }) {
  return (
    <div className={`flex items-center gap-3 ${className}`} aria-hidden="true">
      {labels.map((label, i) => (
        <div key={i} className="flex-1 flex items-center gap-3">
          <div className="flex-1 h-px bg-hairline relative">
            <div className="absolute left-0 top-0 w-px h-1.5 bg-hairline -translate-y-full" />
          </div>
          <span className="text-[10px] font-mono text-ink-muted tracking-wide shrink-0">
            {label}
          </span>
        </div>
      ))}
    </div>
  );
}
