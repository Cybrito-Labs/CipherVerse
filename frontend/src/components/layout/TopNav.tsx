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
        'flex items-center justify-between px-6',
        'border-b border-[#27272A] bg-[#000000]/80 backdrop-blur-xl',
        'transition-all duration-300 ease-out',
        sidebarCollapsed ? 'left-[72px]' : 'left-[260px]'
      )}
    >
      {/* Breadcrumbs */}
      <nav className="flex items-center gap-2 text-sm font-medium tracking-tight">
        {breadcrumbs.map((crumb, idx) => (
          <div key={crumb.path} className="flex items-center gap-2">
            {idx > 0 && (
              <span className="text-[#52525B]">/</span>
            )}
            {idx === breadcrumbs.length - 1 ? (
              <span className="text-[#EDEDED]">{crumb.label}</span>
            ) : (
              <Link
                to={crumb.path}
                className="text-[#A1A1AA] hover:text-[#EDEDED] transition-colors"
              >
                {crumb.label}
              </Link>
            )}
          </div>
        ))}
      </nav>

      {/* Search */}
      <button
        onClick={onSearchOpen}
        className={cn(
          'flex items-center gap-2 px-3 py-1.5 rounded-md',
          'text-sm font-medium text-[#A1A1AA]',
          'border border-[#27272A] hover:border-[#52525B]',
          'bg-[#0A0A0A] hover:bg-[#171717]',
          'transition-colors duration-200 shadow-sm'
        )}
      >
        <Search className="w-4 h-4" />
        <span className="hidden sm:inline mr-2">Search</span>
        <kbd className="hidden sm:inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-[4px] bg-[#171717] text-[11px] font-sans border border-[#27272A] text-[#A1A1AA]">
          <Command className="w-3 h-3" />K
        </kbd>
      </button>
    </header>
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
