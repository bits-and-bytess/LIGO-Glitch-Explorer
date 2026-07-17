import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getLibraryClass } from "../api";

export default function LibraryDetail() {
  const { className } = useParams();
  const [cls, setCls] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getLibraryClass(className).then(setCls).catch((e) => setError(e.message));
  }, [className]);

  if (error) return <p className="text-amber-400">{error}</p>;
  if (!cls) return <p className="text-slate-400">Loading...</p>;

  return (
    <div className="space-y-8">
      <Link to="/library" className="text-sm text-accent">&larr; Back to library</Link>
      <div>
        <h1 className="text-2xl font-semibold">{cls.name.replace(/_/g, " ")}</h1>
        <p className="text-slate-300 mt-2 max-w-2xl">{cls.description}</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
        <Stat label="Physical origin" value={cls.physical_origin || "Unknown / unconfirmed"} />
        <Stat label="Frequency range" value={`${cls.frequency_range_hz[0]}-${cls.frequency_range_hz[1]} Hz`} />
        <Stat label="Typical duration" value={`${cls.typical_duration_s[0]}-${cls.typical_duration_s[1]} s`} />
      </div>

      <div>
        <h2 className="text-lg font-medium mb-3">Examples with GradCAM overlays</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {(cls.all_examples || [{ spectrogram_url: cls.example_spectrogram_url, gradcam_url: cls.example_gradcam_url }]).map((ex, i) => (
            <div key={i} className="bg-panel rounded-lg overflow-hidden border border-slate-800">
              <img src={ex.gradcam_url} alt="" className="w-full aspect-square object-cover" />
            </div>
          ))}
        </div>
      </div>

      {cls.similar_classes?.length > 0 && (
        <div>
          <h2 className="text-lg font-medium mb-2">Often confused with</h2>
          <div className="flex gap-2 flex-wrap">
            {cls.similar_classes.map((s) => (
              <Link key={s} to={`/library/${s}`} className="px-3 py-1 rounded-full bg-white/5 hover:bg-white/10 text-sm">
                {s.replace(/_/g, " ")}
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="bg-panel rounded-lg p-3 border border-slate-800">
      <div className="text-slate-500 text-xs uppercase tracking-wide">{label}</div>
      <div className="mt-1">{value}</div>
    </div>
  );
}
