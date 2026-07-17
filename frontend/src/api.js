// In dev, "/api" is proxied to localhost:8000 by vite.config.js.
// In production (Vercel), set VITE_API_BASE to the deployed backend's
// full URL (e.g. https://your-space.hf.space or https://your-app.onrender.com).
const BASE = import.meta.env.VITE_API_BASE || "/api";

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
