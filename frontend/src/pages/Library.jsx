import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getLibrary, mediaUrl } from "../api";

export default function Library() {
  const [classes, setClasses] = useState(null);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    getLibrary(search).then(setClasses).catch((e) => setError(e.message));
  }, [search]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-semibold">Glitch Library</h1>
          <p className="text-sm text-ink-muted mt-1">22 known classes, each with GradCAM overlays showing what defines it.</p>
        </div>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search classes (e.g. 'blip', 'power line')"
          className="bg-panel border border-hairline rounded-md px-3 py-2 text-sm w-72 font-mono focus:outline-none focus:border-teal/60"
        />
      </div>

      {error && (
        <p className="text-anomaly-bright text-sm font-mono">
          {error} -- have you run <code>scripts/generate_library_assets.py</code> yet?
        </p>
      )}

      {!classes && !error && <p className="text-ink-muted">Loading...</p>}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {classes?.map((c) => (
          <Link
            key={c.name}
            to={`/library/${encodeURIComponent(c.name)}`}
            className="bg-panel rounded-xl overflow-hidden border border-hairline hover:border-teal/50 transition-colors group"
          >
            <img src={mediaUrl(c.example_gradcam_url)} alt={`${c.name} GradCAM example`} className="w-full aspect-square object-cover bg-black" />
            <div className="p-4 space-y-1.5">
              <h3 className="font-display font-medium group-hover:text-teal transition-colors">{c.name.replace(/_/g, " ")}</h3>
              <p className="text-xs text-ink-muted line-clamp-2">{c.description}</p>
              <p className="text-xs font-mono text-ink-muted">
                {c.frequency_range_hz[0]}&ndash;{c.frequency_range_hz[1]} Hz
              </p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
