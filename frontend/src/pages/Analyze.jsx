import { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { analyzeUpload, analyzeGPS, getResult } from "../api";
import OODScoreChart from "../components/OODScoreChart";

const FORMATS = [
  { id: "hdf5", label: "HDF5 strain file" },
  { id: "gps", label: "GPS time + detector" },
  { id: "image", label: "Spectrogram image" },
  { id: "csv", label: "Raw timeseries CSV" },
];
const DURATIONS = [0.5, 1, 2, 4];

export default function Analyze() {
  const { resultId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [format, setFormat] = useState("hdf5");
  const [file, setFile] = useState(location.state?.droppedFile ?? null);
  const [detector, setDetector] = useState("H1");
  const [gpsTime, setGpsTime] = useState("");
  const [duration, setDuration] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (resultId) {
      setLoading(true);
      getResult(resultId)
        .then(setResult)
        .catch((e) => setError(e.message))
        .finally(() => setLoading(false));
    }
  }, [resultId]);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      let res;
      if (format === "gps") {
        const parsedGpsTime = parseFloat(gpsTime);
        if (gpsTime.trim() === "" || !Number.isFinite(parsedGpsTime)) {
          throw new Error("Enter a valid GPS time (e.g. 1369062018).");
        }
        if (parsedGpsTime < 0) {
          throw new Error("GPS time must be non-negative.");
        }
        res = await analyzeGPS({ gpsTime: parsedGpsTime, detector, duration });
      } else {
        if (!file) throw new Error("Choose a file first.");
        res = await analyzeUpload({ file, inputFormat: format, detector, duration });
      }
      setResult(res);
      navigate(`/analyze/${res.result_id}`, { replace: true });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-display font-semibold">Analyze a Signal</h1>
        <p className="text-sm text-ink-muted mt-1">Every result gets a permanent, shareable link.</p>
      </div>

      <form onSubmit={handleSubmit} className="bg-panel border border-hairline rounded-xl p-6 space-y-5 max-w-2xl">
        <div className="flex gap-2 flex-wrap">
          {FORMATS.map((f) => (
            <button
              type="button"
              key={f.id}
              onClick={() => setFormat(f.id)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium border transition-colors ${
                format === f.id ? "border-teal text-teal bg-teal/10" : "border-hairline text-ink-muted hover:text-ink"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {format === "gps" ? (
          <div className="grid grid-cols-2 gap-4">
            <label className="text-sm space-y-1">
              <span className="text-ink-muted">GPS time</span>
              <input
                value={gpsTime}
                onChange={(e) => setGpsTime(e.target.value)}
                placeholder="e.g. 1369062018"
                required
                inputMode="decimal"
                className="w-full bg-void border border-hairline rounded-md px-3 py-2 font-mono focus:outline-none focus:border-teal/60"
              />
            </label>
            <label className="text-sm space-y-1">
              <span className="text-ink-muted">Detector</span>
              <select value={detector} onChange={(e) => setDetector(e.target.value)} className="w-full bg-void border border-hairline rounded-md px-3 py-2 focus:outline-none focus:border-teal/60">
                <option value="H1">H1 -- Hanford</option>
                <option value="L1">L1 -- Livingston</option>
              </select>
            </label>
          </div>
        ) : (
          <div className="space-y-1 text-sm">
            <span className="text-ink-muted">
              {format === "hdf5" && "GWOSC .hdf5/.h5 strain file"}
              {format === "image" && "Spectrogram image (PNG/JPG)"}
              {format === "csv" && "CSV with time + strain columns"}
            </span>
            <input
              type="file"
              accept={format === "image" ? "image/png,image/jpeg" : format === "csv" ? ".csv" : ".h5,.hdf5"}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm text-ink-muted file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border file:border-hairline file:bg-panel-raised file:text-ink file:text-sm"
            />
            {file && <p className="text-xs font-mono text-ink-muted">Selected: {file.name}</p>}
          </div>
        )}

        <label className="text-sm space-y-1 block">
          <span className="text-ink-muted">Duration</span>
          <div className="flex gap-2">
            {DURATIONS.map((d) => (
              <button
                type="button"
                key={d}
                onClick={() => setDuration(d)}
                className={`px-3 py-1 rounded-md text-sm font-mono border transition-colors ${
                  duration === d ? "border-teal text-teal" : "border-hairline text-ink-muted"
                }`}
              >
                {d}s
              </button>
            ))}
          </div>
        </label>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 rounded-md bg-teal hover:bg-teal/90 disabled:opacity-50 font-display font-medium text-void"
        >
          {loading ? "Running inference..." : "Analyze"}
        </button>

        {error && <p className="text-anomaly-bright text-sm font-mono">{error}</p>}
      </form>

      {result && <ResultPanel result={result} />}
    </div>
  );
}

function ResultPanel({ result }) {
  const shareUrl = `${window.location.origin}/analyze/${result.result_id}`;
  const sortedProbs = Object.entries(result.class_probabilities).sort((a, b) => b[1] - a[1]).slice(0, 5);

  return (
    <div className="bg-panel border border-hairline rounded-xl p-6 max-w-2xl space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-display font-medium">
          {result.predicted_class.replace(/_/g, " ")}
          <span className="text-ink-muted text-sm ml-2 font-mono">{(result.confidence * 100).toFixed(1)}%</span>
        </h2>
        {result.ood_flagged && (
          <span className="text-xs px-2 py-1 rounded-full bg-anomaly/15 text-anomaly-bright font-mono font-medium">Flagged as OOD</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs font-mono text-ink-muted mb-1">Q-transform spectrogram</p>
          <img src={result.spectrogram_url} className="rounded-md w-full" />
        </div>
        <div>
          <p className="text-xs font-mono text-ink-muted mb-1">GradCAM overlay</p>
          <img src={result.gradcam_url} className="rounded-md w-full" />
        </div>
      </div>

      <div>
        <p className="text-xs font-mono text-ink-muted mb-2">Top class probabilities</p>
        <div className="space-y-1.5">
          {sortedProbs.map(([name, p]) => (
            <div key={name} className="flex items-center gap-2 text-sm">
              <span className="w-40 truncate">{name.replace(/_/g, " ")}</span>
              <div className="flex-1 h-2 bg-void rounded-full overflow-hidden">
                <div className="h-full bg-teal" style={{ width: `${p * 100}%` }} />
              </div>
              <span className="w-14 text-right text-ink-muted font-mono text-xs">{(p * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-void rounded-md p-3 border border-hairline">
        <p className="text-xs font-mono text-ink-muted mb-1">OOD score: {result.ood_score.toFixed(3)}</p>
        <OODScoreChart score={result.ood_score} threshold={result.ood_threshold} flagged={result.ood_flagged} />
        <p className="text-sm text-ink-muted mt-1">{result.ood_interpretation}</p>
      </div>

      {result.warnings?.length > 0 && (
        <ul className="text-xs font-mono text-anomaly-bright list-disc list-inside space-y-1">
          {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      )}

      <div className="flex items-center gap-2 text-xs font-mono text-ink-muted">
        <span>Shareable link:</span>
        <code className="bg-void px-2 py-1 rounded border border-hairline">{shareUrl}</code>
        <button
          onClick={() => navigator.clipboard.writeText(shareUrl)}
          className="text-teal underline"
        >
          copy
        </button>
      </div>
    </div>
  );
}
