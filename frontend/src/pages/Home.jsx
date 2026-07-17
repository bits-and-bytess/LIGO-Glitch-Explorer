import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const [dragOver, setDragOver] = useState(false);
  const navigate = useNavigate();

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      // Hand off to the Analyze page with the file pre-loaded via
      // navigation state, so the drop-to-analyze quickstart actually works.
      navigate("/analyze", { state: { droppedFile: file } });
    }
  }

  return (
    <div className="space-y-10">
      <section className="text-center max-w-3xl mx-auto space-y-4 py-10">
        <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">
          Understand LIGO's noise, not just its signals
        </h1>
        <p className="text-slate-300 leading-relaxed">
          LIGO Glitch Explorer classifies gravitational-wave detector noise
          ("glitches"), shows exactly which time-frequency features drove
          each classification with GradCAM, and flags signals that don't
          match any known type -- surfacing candidates for entirely new
          glitch classes. Built for LIGO/Virgo/KAGRA scientists and anyone
          curious how detector noise is characterized.
        </p>
        <div className="flex justify-center gap-3 pt-2 text-sm">
          <a href="/library" className="px-4 py-2 rounded-md bg-white/5 hover:bg-white/10">Browse the Glitch Library</a>
          <a href="/gallery" className="px-4 py-2 rounded-md bg-white/5 hover:bg-white/10">See the Anomaly Gallery</a>
        </div>
      </section>

      <section
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`rounded-xl border-2 border-dashed p-12 text-center transition-colors ${
          dragOver ? "border-accent bg-accent/5" : "border-slate-700"
        }`}
      >
        <p className="text-lg font-medium mb-1">Drop a strain file to analyze it now</p>
        <p className="text-sm text-slate-400 mb-4">
          Accepts GWOSC .hdf5 files, spectrogram images, or strain CSVs.
          Or head to the Analyze page to pull data by GPS time instead.
        </p>
        <a href="/analyze" className="text-accent underline text-sm">Or open the full Analyze page &rarr;</a>
      </section>
    </div>
  );
}
