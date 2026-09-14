import { useLocation, Link } from 'react-router-dom';
import { Search, Command, Menu, Sun, Moon } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import { allNavItems } from '@/constants/navigation';

interface TopNavProps {
  sidebarCollapsed: boolean;
  onSearchOpen: () => void;
  onMobileMenuToggle: () => void;
}

export function TopNav({ sidebarCollapsed, onSearchOpen, onMobileMenuToggle }: TopNavProps) {
  const location = useLocation();
  const breadcrumbs = getBreadcrumbs(location.pathname);

  return (
    <header
      className={cn(
        'fixed top-0 right-0 z-30 h-16 left-0',
        'flex items-center justify-between px-3 sm:px-6 gap-2 sm:gap-4',
        'border-b border-border bg-background/80 backdrop-blur-xl',
        'transition-all duration-300 ease-out',
        sidebarCollapsed ? 'md:left-[72px]' : 'md:left-[260px]'
      )}
    >
      {/* Left: Mobile Hamburger Toggle + Breadcrumbs */}
      <div className="flex items-center gap-2 min-w-0 flex-1">
        <button
          onClick={onMobileMenuToggle}
          className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-md transition-colors md:hidden flex-shrink-0"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <nav
          aria-label="Breadcrumb"
          className="flex items-center gap-1.5 text-xs sm:text-sm font-medium tracking-tight overflow-hidden text-ellipsis whitespace-nowrap"
        >
          <ol
            itemScope
            itemType="https://schema.org/BreadcrumbList"
            className="flex items-center gap-1.5 min-w-0"
          >
            {breadcrumbs.map((crumb, idx) => (
              <li
                key={crumb.path}
                itemProp="itemListElement"
                itemScope
                itemType="https://schema.org/ListItem"
                className="flex items-center gap-1.5 min-w-0"
              >
                {idx > 0 && (
                  <span className="text-muted-foreground flex-shrink-0" aria-hidden="true">/</span>
                )}
                {idx === breadcrumbs.length - 1 ? (
                  <span
                    itemProp="name"
                    aria-current="page"
                    className="text-foreground truncate"
                  >
                    {crumb.label}
                  </span>
                ) : (
                  <Link
                    itemProp="item"
                    to={crumb.path}
                    className="text-muted-foreground hover:text-foreground transition-colors truncate hidden sm:inline"
                  >
                    <span itemProp="name">{crumb.label}</span>
                  </Link>
                )}
                <meta itemProp="position" content={String(idx + 1)} />
              </li>
            ))}
          </ol>
        </nav>
      </div>

      {/* Center: Search */}
      <div className="flex-1 flex justify-center max-w-xs sm:max-w-md">
        <button
          onClick={onSearchOpen}
          className={cn(
            'flex items-center gap-2 px-2.5 py-1.5 rounded-md w-full',
            'text-xs sm:text-sm font-medium text-muted-foreground',
            'border border-border hover:border-muted-foreground',
            'bg-card hover:bg-secondary',
            'transition-colors duration-200 shadow-sm'
          )}
        >
          <Search className="w-3.5 h-3.5 sm:w-4 sm:h-4 flex-shrink-0" />
          <span className="text-left truncate flex-1">Search tools...</span>
          <kbd className="hidden sm:inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-[4px] bg-secondary text-[11px] font-sans border border-border text-muted-foreground flex-shrink-0">
            <Command className="w-3 h-3" />K
          </kbd>
        </button>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center justify-end flex-shrink-0">
        <ThemeToggle />
      </div>
    </header>
  );
}

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
