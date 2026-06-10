import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Shield } from 'lucide-react';
import { cn } from '@/lib/utils';
import { navigationGroups } from '@/constants/navigation';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const location = useLocation();

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 72 : 260 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        'fixed left-0 top-0 z-40 h-screen flex flex-col',
        'bg-[#000000] border-r border-[#27272A]'
      )}
    >
      {/* Logo */}
      <div className="flex items-center h-16 px-4 border-b border-[#27272A]">
        <NavLink to="/" className="flex items-center gap-3 min-w-0">
          <div className="flex-shrink-0 w-8 h-8 rounded-md bg-[#EDEDED] flex items-center justify-center">
            <Shield className="w-4 h-4 text-[#000000]" />
          </div>
          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden whitespace-nowrap"
              >
                <span className="text-base font-semibold tracking-tight text-[#EDEDED]">
                  CipherVerse
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </NavLink>
      </div>

      {/* Navigation */}
      <ScrollArea className="flex-1 py-4">
        <nav className="space-y-6 px-3">
          {navigationGroups.map((group) => (
            <div key={group.label}>
              <AnimatePresence>
                {!collapsed && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="px-3 mb-2 text-[11px] font-medium text-[#A1A1AA] uppercase tracking-wider"
                  >
                    {group.label}
                  </motion.p>
                )}
              </AnimatePresence>
              <div className="space-y-0.5">
                {group.items.map((item) => {
                  const isActive =
                    item.path === '/'
                      ? location.pathname === '/'
                      : location.pathname.startsWith(item.path);
                  const Icon = item.icon;

                  const linkContent = (
                    <NavLink
                      to={item.path}
                      className={cn(
                        'flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium',
                        'transition-colors duration-150 group relative',
                        isActive
                          ? 'bg-[#171717] text-[#EDEDED]'
                          : 'text-[#A1A1AA] hover:bg-[#0A0A0A] hover:text-[#EDEDED]'
                      )}
                    >
                      {isActive && (
                        <motion.div
                          layoutId="sidebar-indicator"
                          className="absolute left-0 top-[20%] bottom-[20%] w-[3px] rounded-r-md bg-[#EDEDED]"
                          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                        />
                      )}
                      <Icon
                        className={cn(
                          'w-[16px] h-[16px] flex-shrink-0 transition-colors',
                          isActive ? 'text-[#EDEDED]' : 'text-[#A1A1AA] group-hover:text-[#EDEDED]'
                        )}
                      />
                      <AnimatePresence>
                        {!collapsed && (
                          <motion.div
                            initial={{ opacity: 0, width: 0 }}
                            animate={{ opacity: 1, width: 'auto' }}
                            exit={{ opacity: 0, width: 0 }}
                            transition={{ duration: 0.15 }}
                            className="flex items-center justify-between flex-1 overflow-hidden"
                          >
                            <span className="whitespace-nowrap">{item.label}</span>
                            {item.toolCount && (
                              <Badge
                                variant="secondary"
                                className="ml-auto text-[10px] px-1.5 py-0 h-5 bg-[#27272A] text-[#EDEDED] border-none font-medium"
                              >
                                {item.toolCount}
                              </Badge>
                            )}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </NavLink>
                  );

                  if (collapsed) {
                    return (
                      <Tooltip key={item.path} delayDuration={0}>
                        <TooltipTrigger asChild>{linkContent}</TooltipTrigger>
                        <TooltipContent side="right" sideOffset={12} className="bg-[#0A0A0A] border-[#27272A] text-[#EDEDED]">
                          <p className="font-medium">{item.label}</p>
                          <p className="text-xs text-[#A1A1AA]">{item.description}</p>
                        </TooltipContent>
                      </Tooltip>
                    );
                  }

                  return <div key={item.path}>{linkContent}</div>;
                })}
              </div>
            </div>
          ))}
        </nav>
      </ScrollArea>

      {/* Collapse Toggle */}
      <div className="border-t border-[#27272A] p-3">
        <button
          onClick={onToggle}
          className={cn(
            'flex items-center justify-center w-full py-2 rounded-md',
            'text-[#A1A1AA] hover:text-[#EDEDED] hover:bg-[#171717]',
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
    </motion.aside>
  );
}
