import { Outlet, NavLink } from "react-router-dom";

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
      <header className="border-b border-slate-800 bg-panel/60 backdrop-blur sticky top-0 z-10">
        <nav className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <NavLink to="/" className="font-semibold tracking-tight text-lg">
            LIGO Glitch <span className="text-accent">Explorer</span>
          </NavLink>
          <div className="flex gap-1 text-sm">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-md transition-colors ${
                    isActive ? "bg-accent/15 text-accent" : "text-slate-300 hover:text-white hover:bg-white/5"
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
      <footer className="border-t border-slate-800 text-xs text-slate-500 py-6 text-center">
        Built on GWOSC open data and the Gravity Spy dataset. Not for
        detection-confidence decisions -- a research/education tool.
      </footer>
    </div>
  );
}
