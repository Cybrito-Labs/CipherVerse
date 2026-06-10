import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import type { LucideIcon } from 'lucide-react';
import { ToolHeader } from './ToolHeader';

interface ToolPageLayoutProps {
  title: string;
  description: string;
  icon: LucideIcon;
  badges?: { label: string; variant?: 'default' | 'success' | 'warning' }[];
  children: ReactNode;
  className?: string;
  // If true, the children won't be wrapped in the 45/55 grid layout (for custom pages like Dashboard)
  noGrid?: boolean; 
}

export function ToolPageLayout({
  title,
  description,
  icon,
  badges,
  children,
  className,
  noGrid = false,
}: ToolPageLayoutProps) {
  return (
    <div className={cn('max-w-[1200px] mx-auto space-y-6 pb-12', className)}>
      <ToolHeader 
        title={title} 
        description={description} 
        icon={icon} 
        badges={badges} 
      />
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.05, ease: [0.16, 1, 0.3, 1] }}
        className={cn(
          !noGrid && "grid grid-cols-1 lg:grid-cols-[45%_55%] gap-6 items-start"
        )}
      >
        {children}
      </motion.div>
    </div>
  );
}
