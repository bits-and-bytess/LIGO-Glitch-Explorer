import { Outlet, NavLink } from "react-router-dom";
import ChirpMark from "./ChirpMark";

const navItems = [
  { to: "/", label: "Home", end: true },
  { to: "/library", label: "Glitch Library" },
  { to: "/analyze", label: "Analyze a Signal" },
  { to: "/gallery", label: "Anomaly Gallery" },
  { to: "/methodology", label: "Methodology" },
];

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-hairline bg-panel/70 backdrop-blur sticky top-0 z-10">
        <nav className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <NavLink to="/" className="flex items-center gap-2.5 font-display font-semibold tracking-tight text-lg">
            <ChirpMark className="w-7 h-5 text-teal" />
            <span>
              LIGO Glitch <span className="text-teal">Explorer</span>
            </span>
          </NavLink>
          <div className="flex gap-1 text-sm font-medium">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-md transition-colors ${
                    isActive ? "bg-teal/15 text-teal" : "text-ink-muted hover:text-ink hover:bg-white/5"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        </nav>
      </header>
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8">
        <Outlet />
      </main>
      <footer className="border-t border-hairline text-xs text-ink-muted py-6 text-center font-mono">
        Built on GWOSC open data and the Gravity Spy dataset. Not for
        detection-confidence decisions -- a research/education tool.
      </footer>
    </div>
  );
}
