import { useLocation, Link } from 'react-router-dom';
import { Search, Command } from 'lucide-react';
import { cn } from '@/lib/utils';
import { allNavItems } from '@/constants/navigation';

interface TopNavProps {
  sidebarCollapsed: boolean;
  onSearchOpen: () => void;
}

export function TopNav({ sidebarCollapsed, onSearchOpen }: TopNavProps) {
  const location = useLocation();

  const breadcrumbs = getBreadcrumbs(location.pathname);

  return (
    <header
      className={cn(
        'fixed top-0 right-0 z-30 h-16',
        'flex items-center justify-between px-6 gap-4',
        'border-b border-border bg-background/80 backdrop-blur-xl',
        'transition-all duration-300 ease-out',
        sidebarCollapsed ? 'left-[72px]' : 'left-[260px]'
      )}
    >
      {/* Left: Breadcrumbs */}
      <nav className="flex items-center gap-2 text-sm font-medium tracking-tight flex-1">
        {breadcrumbs.map((crumb, idx) => (
          <div key={crumb.path} className="flex items-center gap-2">
            {idx > 0 && (
              <span className="text-muted-foreground">/</span>
            )}
            {idx === breadcrumbs.length - 1 ? (
              <span className="text-foreground">{crumb.label}</span>
            ) : (
              <Link
                to={crumb.path}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                {crumb.label}
              </Link>
            )}
          </div>
        ))}
      </nav>

      {/* Center: Search */}
      <div className="flex-1 flex justify-center">
        <button
          onClick={onSearchOpen}
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded-md w-full max-w-md',
            'text-sm font-medium text-muted-foreground',
            'border border-border hover:border-muted-foreground',
            'bg-card hover:bg-secondary',
            'transition-colors duration-200 shadow-sm'
          )}
        >
          <Search className="w-4 h-4" />
          <span className="hidden sm:inline flex-1 text-left">Search documentation or tools...</span>
          <kbd className="hidden sm:inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-[4px] bg-secondary text-[11px] font-sans border border-border text-muted-foreground">
            <Command className="w-3 h-3" />K
          </kbd>
        </button>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center justify-end flex-1">
        <ThemeToggle />
      </div>
    </header>
  );
}

import { useTheme } from 'next-themes';
import { Sun, Moon } from 'lucide-react';
import { useEffect, useState } from 'react';

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-md transition-colors w-9 h-9">
        <span className="sr-only">Toggle theme</span>
      </button>
    );
  }

  return (
    <button
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
      className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-md transition-colors w-9 h-9 flex items-center justify-center"
      title="Toggle Theme"
    >
      {theme === 'dark' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
      <span className="sr-only">Toggle theme</span>
    </button>
  );
}

function getBreadcrumbs(pathname: string) {
  const crumbs: { label: string; path: string }[] = [
    { label: 'CipherVerse', path: '/' },
  ];

  if (pathname === '/') return crumbs;

  const navItem = allNavItems.find(
    (item) =>
      item.path === pathname || pathname.startsWith(item.path + '/')
  );

  if (navItem) {
    crumbs.push({ label: navItem.label, path: navItem.path });
  }

  // Handle sub-paths (e.g., /classical/caesar)
  const segments = pathname.split('/').filter(Boolean);
  if (segments.length > 1) {
    const subLabel = segments[segments.length - 1]
      .replace(/-/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());
    crumbs.push({ label: subLabel, path: pathname });
  }

  return crumbs;
}
