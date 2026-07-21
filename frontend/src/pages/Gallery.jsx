import { useEffect, useState } from "react";
import { getGallery, mediaUrl } from "../api";

export default function Gallery() {
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState(null);
  const [detector, setDetector] = useState("");
  const [minScore, setMinScore] = useState("");

  useEffect(() => {
    getGallery({ detector: detector || undefined, min_ood_score: minScore || undefined })
      .then(setEntries)
      .catch((e) => setError(e.message));
  }, [detector, minScore]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display font-semibold">Anomaly Gallery</h1>
        <p className="text-ink-muted text-sm mt-1 max-w-2xl leading-relaxed">
          O4 signals the model couldn't confidently match to any known
          Gravity Spy class, sorted by OOD score. Candidates for new
          glitch types -- not confirmed detections of anything.
        </p>
      </div>

      <div className="flex gap-3 items-center text-sm">
        <select value={detector} onChange={(e) => setDetector(e.target.value)} className="bg-panel border border-hairline rounded-md px-3 py-1.5 focus:outline-none focus:border-anomaly/60">
          <option value="">All detectors</option>
          <option value="H1">H1</option>
          <option value="L1">L1</option>
        </select>
        <input
          type="number"
          step="0.1"
          value={minScore}
          onChange={(e) => setMinScore(e.target.value)}
          placeholder="Min OOD score"
          className="bg-panel border border-hairline rounded-md px-3 py-1.5 w-36 font-mono focus:outline-none focus:border-anomaly/60"
        />
      </div>

      {error && (
        <p className="text-anomaly-bright text-sm font-mono">
          {error} -- have you run <code>scripts/curate_anomaly_gallery.py</code> yet?
        </p>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {entries?.map((e) => (
          <div key={e.id} className="bg-panel rounded-xl overflow-hidden border border-hairline hover:border-anomaly/40 transition-colors">
            <img src={mediaUrl(e.gradcam_url)} alt="" className="w-full aspect-square object-cover bg-black" />
            <div className="p-4 space-y-1 text-sm">
              <div className="flex justify-between font-mono">
                <span className="font-medium">{e.detector}</span>
                <span className="text-anomaly-bright">OOD {e.ood_score.toFixed(2)}</span>
              </div>
              <p className="text-xs font-mono text-ink-muted">GPS {e.gps_time}</p>
              <p className="text-xs text-ink-muted">{e.why_flagged}</p>
              {e.nearest_known_class && (
                <p className="text-xs font-mono text-ink-muted">Nearest known: {e.nearest_known_class.replace(/_/g, " ")}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
