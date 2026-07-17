/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        void: "#0b0e14",
        panel: "#131826",
        accent: "#5eead4",
        signal: "#818cf8",
      },
    },
  },
  plugins: [],
}
