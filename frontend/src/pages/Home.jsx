import { useState } from "react";
import { useNavigate } from "react-router-dom";
import ChirpMark from "../components/ChirpMark";
import FrequencyAxis from "../components/FrequencyAxis";

export default function Home() {
  const [dragOver, setDragOver] = useState(false);
  const navigate = useNavigate();

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      navigate("/analyze", { state: { droppedFile: file } });
    }
  }

  return (
    <div className="space-y-14">
      <section className="max-w-3xl mx-auto space-y-6 py-8">
        <div className="flex items-center gap-3 text-teal">
          <ChirpMark className="w-10 h-6" />
          <span className="font-mono text-xs tracking-widest uppercase text-ink-muted">
            H1 / L1 &middot; O4 public data
          </span>
        </div>
        <h1 className="text-3xl md:text-4xl font-display font-semibold tracking-tight leading-tight">
          Every glitch has a shape.
          <br />
          <span className="text-ink-muted">This is where you learn to read it.</span>
        </h1>
        <p className="text-ink-muted leading-relaxed max-w-xl">
          LIGO Glitch Explorer classifies gravitational-wave detector noise,
          shows exactly which time-frequency features drove each call with
          GradCAM, and flags signals that match nothing on record --
          candidates for a glitch type nobody's named yet. Built for anyone
          who reads Q-transforms for a living.
        </p>
        <div className="flex gap-3 pt-1 text-sm font-medium">
          <a href="/library" className="px-4 py-2 rounded-md bg-panel-raised border border-hairline hover:border-teal/50 transition-colors">
            Browse the Glitch Library
          </a>
          <a href="/gallery" className="px-4 py-2 rounded-md bg-panel-raised border border-hairline hover:border-anomaly/50 transition-colors">
            See the Anomaly Gallery
          </a>
        </div>
      </section>

      <FrequencyAxis />

      <section
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`rounded-xl border-2 border-dashed p-12 text-center transition-colors ${
          dragOver ? "border-teal bg-teal/5" : "border-hairline"
        }`}
      >
        <p className="text-lg font-display font-medium mb-1">Drop a strain file to analyze it now</p>
        <p className="text-sm text-ink-muted mb-4">
          Accepts GWOSC .hdf5 files, spectrogram images, or strain CSVs.
          Or open the Analyze page to pull data by GPS time instead.
        </p>
        <a href="/analyze" className="text-teal underline text-sm font-medium">Open the full Analyze page &rarr;</a>
      </section>
    </div>
  );
}
