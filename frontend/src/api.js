// In dev, "/api" is proxied to localhost:8000 by vite.config.js.
// In production (Vercel), set VITE_API_BASE to the deployed backend's
// full URL (e.g. https://your-space.hf.space or https://your-app.onrender.com).
const BASE = import.meta.env.VITE_API_BASE || "/api";

// Images/results returned by the API come back as bare paths like
// "/static/results/<id>/spectrogram.png" -- these are served by the
// BACKEND directly (backend/main.py mounts StaticFiles at /static), not
// under the "/api" prefix used for JSON endpoints above. A browser
// resolves a bare "/static/..." src against the CURRENT PAGE's origin
// (the Vite dev server, or wherever the frontend is deployed), not the
// backend's -- so these need their own resolution, separate from BASE.
//
// Dev: MEDIA_ORIGIN is "", so mediaUrl() returns the path unchanged;
// vite.config.js has a matching `/static` proxy rule that forwards it
// to localhost:8000, same as it does for `/api`.
// Production: MEDIA_ORIGIN is the real backend origin from
// VITE_API_BASE, so mediaUrl() builds a full absolute URL that works
// regardless of what origin the frontend itself is served from.
const MEDIA_ORIGIN = import.meta.env.VITE_API_BASE || "";

export function mediaUrl(path) {
  if (!path) return path;
  return `${MEDIA_ORIGIN}${path}`;
}

export async function getLibrary(search) {
  const url = new URL(`${BASE}/library`, window.location.origin);
  if (search) url.searchParams.set("search", search);
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to load glitch library");
  return res.json();
}

export async function getLibraryClass(className) {
  const res = await fetch(`${BASE}/library/${encodeURIComponent(className)}`);
  if (!res.ok) throw new Error("Failed to load glitch class");
  return res.json();
}

export async function analyzeUpload({ file, inputFormat, detector, duration }) {
  const form = new FormData();
  form.append("file", file);
  form.append("input_format", inputFormat);
  if (detector) form.append("detector", detector);
  form.append("duration", duration ?? 1.0);

  const res = await fetch(`${BASE}/analyze/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Analysis failed");
  return res.json();
}

export async function analyzeGPS({ gpsTime, detector, duration }) {
  const form = new FormData();
  form.append("gps_time", gpsTime);
  form.append("detector", detector);
  form.append("duration", duration ?? 1.0);

  const res = await fetch(`${BASE}/analyze/gps`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Analysis failed");
  return res.json();
}

export async function getResult(resultId) {
  const res = await fetch(`${BASE}/analyze/${resultId}`);
  if (!res.ok) throw new Error("Result not found");
  return res.json();
}

export async function getGallery(filters = {}) {
  const url = new URL(`${BASE}/gallery`, window.location.origin);
  Object.entries(filters).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
  });
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to load anomaly gallery");
  return res.json();
}
