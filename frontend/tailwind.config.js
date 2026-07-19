/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        void: "#0c0a16",
        panel: "#17131f",
        "panel-raised": "#201a2c",
        hairline: "#2c2438",
        ink: "#ede9f5",
        "ink-muted": "#948da3",
        teal: "#21918c",
        green: "#35b779",
        yellow: "#fde725",
        anomaly: "#e8823c",
        "anomaly-bright": "#f4a261",
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        sans: ["IBM Plex Sans", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
    },
  },
  plugins: [],
}
