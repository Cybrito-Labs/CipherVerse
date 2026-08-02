import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, ChevronDown, X } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { navigationGroups } from '@/constants/navigation';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  mobileOpen: boolean;
  onMobileClose: () => void;
}

export function Sidebar({ collapsed, onToggle, mobileOpen, onMobileClose }: SidebarProps) {
  const location = useLocation();

  return (
    <>
      {/* Mobile Backdrop */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onMobileClose}
            role="button"
            aria-label="Close mobile navigation overlay"
            className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm md:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar Navigation Drawer */}
      <aside
        className={cn(
          'fixed left-0 top-0 z-50 h-screen flex flex-col',
          'bg-background border-r border-border',
          'transition-all duration-300 ease-in-out',
          // Mobile responsive drawer positioning
          mobileOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full md:translate-x-0'
        )}
        style={{
          width: undefined, // Handled by CSS on mobile, style on desktop
        }}
      >
        {/* Animated width wrapper for desktop */}
        <div
          className="h-full flex flex-col transition-all duration-300 ease-out overflow-hidden"
          style={{
            width: typeof window !== 'undefined' && window.innerWidth < 768
              ? '280px'
              : collapsed ? '72px' : '260px'
          }}
        >
          {/* Logo & Mobile Close */}
          <div className="flex items-center justify-between h-16 px-4 border-b border-border flex-shrink-0">
            <NavLink to="/" onClick={onMobileClose} className="flex items-center gap-3 min-w-0" aria-label="CipherVerse Home">
              <div className="flex-shrink-0 w-8 h-8 rounded-md overflow-hidden flex items-center justify-center">
                <img src="/logo.png" alt="CipherVerse Logo" className="w-full h-full object-cover" />
              </div>
              <span className={cn(
                "font-bold text-base tracking-tight text-foreground transition-opacity duration-200",
                collapsed ? "md:hidden" : "block"
              )}>
                CipherVerse
              </span>
            </NavLink>
            
            {/* Mobile Close Button */}
            <button
              onClick={onMobileClose}
              className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary md:hidden"
              aria-label="Close navigation menu"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Navigation Items */}
          <div className="flex-1 py-4 overflow-y-auto">
            <nav className="space-y-6 px-3 pb-4" aria-label="Main Navigation">
              {navigationGroups.map((group) => (
                <NavGroup
                  key={group.label}
                  group={group}
                  collapsed={collapsed}
                  location={location}
                  onMobileClose={onMobileClose}
                />
              ))}
            </nav>
          </div>

          {/* Desktop Collapse Toggle */}
          <div className="border-t border-border p-3 hidden md:block flex-shrink-0">
            <button
              onClick={onToggle}
              aria-label={collapsed ? "Expand sidebar navigation" : "Collapse sidebar navigation"}
              className={cn(
                'flex items-center justify-center w-full py-2 rounded-md',
                'text-muted-foreground hover:text-foreground hover:bg-secondary',
                'transition-colors duration-150'
              )}
            >
              {collapsed ? (
                <ChevronRight className="w-4 h-4" />
              ) : (
                <ChevronLeft className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}

interface NavItemType {
  path: string;
  label: string;
  description?: string;
  icon: LucideIcon;
  toolCount?: number;
}

interface NavGroupType {
  label: string;
  items: NavItemType[];
}

function NavGroup({
  group,
  collapsed,
  location,
  onMobileClose,
}: {
  group: NavGroupType;
  collapsed: boolean;
  location: ReturnType<typeof useLocation>;
  onMobileClose: () => void;
}) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div>
      <AnimatePresence>
        {(!collapsed || (typeof window !== 'undefined' && window.innerWidth < 768)) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center justify-between px-3 mb-2 cursor-pointer group/label"
            onClick={() => setExpanded(!expanded)}
          >
            <p className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider group-hover/label:text-foreground transition-colors">
              {group.label}
            </p>
            <ChevronDown className={cn("w-3.5 h-3.5 text-muted-foreground transition-transform duration-200", expanded ? "" : "-rotate-90")} />
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence initial={false}>
        {(expanded || collapsed) && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="space-y-0.5 overflow-hidden"
          >
            {group.items.map((item: NavItemType) => {
              const isActive =
                item.path === '/'
                  ? location.pathname === '/'
                  : location.pathname.startsWith(item.path);
              const Icon = item.icon;

              const linkContent = (
                <NavLink
                  to={item.path}
                  onClick={onMobileClose}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium',
                    'transition-colors duration-150 group relative',
                    isActive
                      ? 'bg-secondary text-foreground'
                      : 'text-muted-foreground hover:bg-card hover:text-foreground'
                  )}
                >
                  {isActive && (
                    <motion.div
                      layoutId="sidebar-indicator"
                      className="absolute left-0 top-[20%] bottom-[20%] w-[3px] rounded-r-md bg-foreground"
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    />
                  )}
                  <Icon
                    className={cn(
                      'w-[16px] h-[16px] flex-shrink-0 transition-colors',
                      isActive ? 'text-foreground' : 'text-muted-foreground group-hover:text-foreground'
                    )}
                  />
                  <div className="flex items-center justify-between flex-1 overflow-hidden">
                    <span className="whitespace-nowrap">{item.label}</span>
                    {item.toolCount && (
                      <Badge
                        variant="secondary"
                        className="ml-auto text-[10px] px-1.5 py-0 h-5 bg-input text-foreground border-none font-medium"
                      >
                        {item.toolCount}
                      </Badge>
                    )}
                  </div>
                </NavLink>
              );

              if (collapsed && (typeof window === 'undefined' || window.innerWidth >= 768)) {
                return (
                  <Tooltip key={item.path} delayDuration={0}>
                    <TooltipTrigger asChild>{linkContent}</TooltipTrigger>
                    <TooltipContent side="right" sideOffset={12} className="bg-card border-border text-foreground">
                      <p className="font-medium">{item.label}</p>
                      <p className="text-xs text-muted-foreground">{item.description}</p>
                    </TooltipContent>
                  </Tooltip>
                );
              }

              return <div key={item.path}>{linkContent}</div>;
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
