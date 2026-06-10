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
        'border-b border-[#27272A] bg-[#000000]/80 backdrop-blur-xl',
        'transition-all duration-300 ease-out',
        sidebarCollapsed ? 'left-[72px]' : 'left-[260px]'
      )}
    >
      {/* Left: Breadcrumbs */}
      <nav className="flex items-center gap-2 text-sm font-medium tracking-tight flex-1">
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

      {/* Center: Search */}
      <div className="flex-1 flex justify-center">
        <button
          onClick={onSearchOpen}
          className={cn(
            'flex items-center gap-2 px-3 py-1.5 rounded-md w-full max-w-md',
            'text-sm font-medium text-[#A1A1AA]',
            'border border-[#27272A] hover:border-[#52525B]',
            'bg-[#0A0A0A] hover:bg-[#171717]',
            'transition-colors duration-200 shadow-sm'
          )}
        >
          <Search className="w-4 h-4" />
          <span className="hidden sm:inline flex-1 text-left">Search documentation or tools...</span>
          <kbd className="hidden sm:inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-[4px] bg-[#171717] text-[11px] font-sans border border-[#27272A] text-[#A1A1AA]">
            <Command className="w-3 h-3" />K
          </kbd>
        </button>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center justify-end gap-3 flex-1">
        <button className="p-2 text-[#A1A1AA] hover:text-[#EDEDED] hover:bg-[#171717] rounded-md transition-colors" title="Recent Tools">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        </button>
        <button className="p-2 text-[#A1A1AA] hover:text-[#EDEDED] hover:bg-[#171717] rounded-md transition-colors" title="Toggle Theme">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>
        </button>
        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 border border-[#27272A] flex items-center justify-center text-xs font-bold text-white shadow-sm cursor-pointer ml-1">
          U
        </div>
      </div>
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
