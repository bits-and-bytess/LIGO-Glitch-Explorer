import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getLibraryClass, mediaUrl } from "../api";

export default function LibraryDetail() {
  const { className } = useParams();
  const [cls, setCls] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getLibraryClass(className).then(setCls).catch((e) => setError(e.message));
  }, [className]);

  if (error) return <p className="text-anomaly-bright font-mono">{error}</p>;
  if (!cls) return <p className="text-ink-muted">Loading...</p>;

  return (
    <div className="space-y-8">
      <Link to="/library" className="text-sm text-teal font-medium">&larr; Back to library</Link>
      <div>
        <h1 className="text-2xl font-display font-semibold">{cls.name.replace(/_/g, " ")}</h1>
        <p className="text-ink-muted mt-2 max-w-2xl leading-relaxed">{cls.description}</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
        <Stat label="Physical origin" value={cls.physical_origin || "Unknown / unconfirmed"} />
        <Stat label="Frequency range" value={`${cls.frequency_range_hz[0]}\u2013${cls.frequency_range_hz[1]} Hz`} mono />
        <Stat label="Typical duration" value={`${cls.typical_duration_s[0]}\u2013${cls.typical_duration_s[1]} s`} mono />
      </div>

      <div>
        <h2 className="text-lg font-display font-medium mb-3">Examples with GradCAM overlays</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {(cls.all_examples || [{ spectrogram_url: cls.example_spectrogram_url, gradcam_url: cls.example_gradcam_url }]).map((ex, i) => (
            <div key={i} className="bg-panel rounded-lg overflow-hidden border border-hairline">
              <img src={mediaUrl(ex.gradcam_url)} alt="" className="w-full aspect-square object-cover" />
            </div>
          ))}
        </div>
      </div>

      {cls.similar_classes?.length > 0 && (
        <div>
          <h2 className="text-lg font-display font-medium mb-2">Often confused with</h2>
          <div className="flex gap-2 flex-wrap">
            {cls.similar_classes.map((s) => (
              <Link key={s} to={`/library/${s}`} className="px-3 py-1 rounded-full bg-panel-raised border border-hairline hover:border-teal/50 text-sm transition-colors">
                {s.replace(/_/g, " ")}
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, mono }) {
  return (
    <div className="bg-panel rounded-lg p-3 border border-hairline">
      <div className="text-ink-muted text-xs uppercase tracking-wide font-mono">{label}</div>
      <div className={`mt-1 ${mono ? "font-mono" : ""}`}>{value}</div>
    </div>
  );
}
