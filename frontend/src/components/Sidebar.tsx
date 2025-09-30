import { useState } from "react";
import { Home, History, CreditCard, HelpCircle, ChevronLeft, ChevronRight, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

interface SidebarProps {
  className?: string;
}

const navigation = [
  { name: "Dashboard", icon: Home, href: "#", active: true },
  { name: "History", icon: History, href: "#", active: false },
  { name: "Plans", icon: CreditCard, href: "#", active: false },
  { name: "Help", icon: HelpCircle, href: "#", active: false },
];

export function Sidebar({ className }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div
      className={cn(
        "flex h-screen flex-col bg-sidebar border-r border-sidebar-border transition-all duration-500 ease-smooth relative",
        collapsed ? "w-16" : "w-72",
        className
      )}
    >
      {/* Glow Effect */}
      <div className="absolute inset-0 bg-gradient-glow opacity-30 blur-3xl" />
      
      {/* Header */}
      <div className="relative flex items-center justify-between p-6 border-b border-sidebar-border/50">
        {!collapsed && (
          <div className="flex items-center space-x-3 animate-fade-in">
            <div className="relative">
              <div className="w-10 h-10 bg-gradient-primary rounded-xl flex items-center justify-center shadow-neural">
                <Zap className="h-5 w-5 text-primary-foreground" />
              </div>
              <div className="absolute inset-0 bg-gradient-primary rounded-xl animate-pulse-ring opacity-20" />
            </div>
            <div>
              <span className="text-sidebar-foreground font-display font-bold text-lg">SafeSpikr</span>
              <div className="text-xs text-sidebar-foreground/60 font-medium">AI POWERED</div>
            </div>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-2 rounded-xl hover:bg-sidebar-accent text-sidebar-foreground transition-all duration-300 hover:shadow-glow hover:scale-110"
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="relative flex-1 p-6 space-y-3">
        {navigation.map((item, index) => (
          <a
            key={item.name}
            href={item.href}
            className={cn(
              "group flex items-center px-4 py-3 rounded-xl text-sm font-medium transition-all duration-300 relative overflow-hidden",
              "hover:shadow-neural hover:-translate-y-0.5",
              item.active
                ? "bg-gradient-secondary text-sidebar-primary-foreground shadow-neural"
                : "text-sidebar-foreground hover:bg-sidebar-accent/70 hover:text-sidebar-accent-foreground"
            )}
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            {item.active && (
              <div className="absolute inset-0 bg-gradient-primary opacity-20 rounded-xl" />
            )}
            <item.icon className={cn(
              "h-5 w-5 transition-all duration-300 group-hover:scale-110",
              !collapsed && "mr-3"
            )} />
            {!collapsed && (
              <span className="font-medium animate-fade-in">{item.name}</span>
            )}
            {item.active && !collapsed && (
              <div className="ml-auto w-2 h-2 bg-accent rounded-full animate-glow" />
            )}
          </a>
        ))}
      </nav>

      {/* Footer */}
      {!collapsed && (
        <div className="relative p-6 border-t border-sidebar-border/50">
          <div className="flex items-center space-x-3 p-3 rounded-xl bg-sidebar-accent/30 backdrop-blur-sm hover-float">
            <div className="relative">
              <div className="w-10 h-10 bg-gradient-secondary rounded-full flex items-center justify-center">
                <span className="text-accent-foreground text-sm font-bold">U</span>
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-sidebar-background animate-glow" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-sidebar-foreground truncate">User</p>
              <p className="text-xs text-sidebar-foreground/60 truncate">Premium Plan</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}