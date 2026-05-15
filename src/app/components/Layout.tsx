import { Outlet, NavLink } from "react-router";
import { LayoutDashboard, FolderSearch, Brain, BarChart3, Info } from "lucide-react";

export function Layout() {
  const navItems = [
    { to: "/", icon: LayoutDashboard, label: "Dashboard" },
    { to: "/directory", icon: FolderSearch, label: "Stock Directory" },
    { to: "/workspace", icon: Brain, label: "Prediction Workspace" },
    { to: "/results", icon: BarChart3, label: "Results" },
    { to: "/about", icon: Info, label: "About" },
  ];

  return (
    <div className="flex h-screen bg-[#0a0e1a] text-gray-100">
      <aside className="w-64 bg-[#0d1117] border-r border-gray-800 flex flex-col">
        <div className="p-6 border-b border-gray-800">
          <h1 className="text-xl text-white mb-1">Stock Price Prediction</h1>
          <p className="text-sm text-gray-400">LSTM AI System</p>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                  isActive
                    ? "bg-[#1e3a5f] text-[#60a5fa]"
                    : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                }`
              }
            >
              <item.icon className="w-5 h-5" />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-800">
          <div className="bg-gray-800/50 rounded-lg p-3">
            <p className="text-xs text-gray-400 mb-1">Model Status</p>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-300">LSTM Ready</span>
            </div>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
