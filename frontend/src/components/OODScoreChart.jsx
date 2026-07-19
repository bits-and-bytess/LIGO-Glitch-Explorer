import { ComposedChart, XAxis, YAxis, ReferenceArea, ReferenceLine, Scatter, ResponsiveContainer } from "recharts";

// Shows exactly what the API returns: the calibrated OOD threshold and
// where this specific signal's energy score falls relative to it. No
// fabricated distribution -- just the two real numbers the model
// actually produced, positioned honestly on a shared number line.
export default function OODScoreChart({ score, threshold, flagged }) {
  const span = Math.max(Math.abs(score - threshold), 1) * 1.6;
  const domainMin = Math.min(score, threshold) - span * 0.3;
  const domainMax = Math.max(score, threshold) + span * 0.3;

  const data = [{ x: score, y: 0 }];
  const dotColor = flagged ? "#e8823c" : "#21918c";

  return (
    <div className="w-full">
      <div className="w-full h-16">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 16, right: 20, bottom: 4, left: 20 }}>
            <XAxis type="number" dataKey="x" domain={[domainMin, domainMax]} hide />
            <YAxis type="number" dataKey="y" domain={[-1, 1]} hide />

            <ReferenceArea x1={domainMin} x2={threshold} y1={-1} y2={1} fill="#21918c" fillOpacity={0.08} />
            <ReferenceArea x1={threshold} x2={domainMax} y1={-1} y2={1} fill="#e8823c" fillOpacity={0.08} />

            <ReferenceLine
              x={threshold}
              stroke="#948da3"
              strokeDasharray="3 3"
              label={{
                value: `threshold ${threshold.toFixed(2)}`,
                position: "top",
                fill: "#948da3",
                fontSize: 10,
                fontFamily: "IBM Plex Mono, monospace",
              }}
            />

            <Scatter
              data={data}
              dataKey="y"
              shape={(props) => (
                <circle cx={props.cx} cy={props.cy} r={6} fill={dotColor} stroke="#0c0a16" strokeWidth={2} />
              )}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-between text-[10px] font-mono text-ink-muted px-5 -mt-1">
        <span>in-distribution</span>
        <span>out-of-distribution</span>
      </div>
    </div>
  );
}
