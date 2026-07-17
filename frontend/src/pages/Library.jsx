import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getLibrary } from "../api";

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
        <h1 className="text-2xl font-semibold">Glitch Library</h1>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search classes (e.g. 'blip', 'power line')"
          className="bg-panel border border-slate-700 rounded-md px-3 py-2 text-sm w-72"
        />
      </div>

      {error && (
        <p className="text-amber-400 text-sm">
          {error} -- have you run <code>scripts/generate_library_assets.py</code> yet?
        </p>
      )}

      {!classes && !error && <p className="text-slate-400">Loading...</p>}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {classes?.map((c) => (
          <Link
            key={c.name}
            to={`/library/${encodeURIComponent(c.name)}`}
            className="bg-panel rounded-xl overflow-hidden border border-slate-800 hover:border-accent/50 transition-colors"
          >
            <img src={c.example_gradcam_url} alt={`${c.name} GradCAM example`} className="w-full aspect-square object-cover bg-black" />
            <div className="p-4 space-y-1">
              <h3 className="font-medium">{c.name.replace(/_/g, " ")}</h3>
              <p className="text-xs text-slate-400 line-clamp-2">{c.description}</p>
              <p className="text-xs text-slate-500">
                {c.frequency_range_hz[0]}-{c.frequency_range_hz[1]} Hz
              </p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
