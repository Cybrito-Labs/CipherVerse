import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import type { LucideIcon } from 'lucide-react';

interface ToolPageLayoutProps {
  title: string;
  description: string;
  icon: LucideIcon;
  children: ReactNode;
  className?: string;
}

export function ToolPageLayout({
  title,
  description,
  icon: Icon,
  children,
  className,
}: ToolPageLayoutProps) {
  return (
    <div className={cn('max-w-[1200px] mx-auto space-y-8 pb-12', className)}>
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="flex items-start gap-5 border-b border-[#27272A] pb-6"
      >
        <div className="w-12 h-12 rounded-lg bg-[#0A0A0A] border border-[#27272A] flex items-center justify-center flex-shrink-0 shadow-sm">
          <Icon className="w-5 h-5 text-[#EDEDED]" />
        </div>
        <div className="pt-0.5">
          <h1 className="text-2xl font-semibold text-[#EDEDED] tracking-tight">{title}</h1>
          <p className="text-[15px] text-[#A1A1AA] mt-1.5 max-w-3xl leading-relaxed">{description}</p>
        </div>
      </motion.div>

      {/* Content */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.05, ease: [0.16, 1, 0.3, 1] }}
      >
        {children}
      </motion.div>
    </div>
  );
}
