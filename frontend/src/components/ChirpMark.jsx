// The site's signature mark: a rising-frequency, rising-amplitude sine
// sweep -- the same shape as the actual GW150914 chirp waveform, one of
// the most recognizable images in gravitational-wave physics. Used
// small and sparingly (nav mark, loading/empty states), not decoratively
// repeated elsewhere.
export default function ChirpMark({ className = "w-6 h-6", stroke = "currentColor" }) {
  // Frequency and amplitude both increase left to right, same qualitative
  // shape as a compact-binary-coalescence chirp.
  const points = [];
  const n = 120;
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const freq = 1.5 + t * 10;
    const amp = 2 + t * 7;
    const x = t * 100;
    const y = 20 - amp * Math.sin(t * freq * Math.PI * 2);
    points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  return (
    <svg viewBox="0 0 100 40" className={className} fill="none" xmlns="http://www.w3.org/2000/svg">
      <polyline
        points={points.join(" ")}
        stroke={stroke}
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
