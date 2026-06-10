import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface ToolInputPanelProps {
  children: ReactNode;
  className?: string;
}

export function ToolInputPanel({ children, className }: ToolInputPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "bg-[#0A0A0A] border border-[#27272A] rounded-[14px] p-6 shadow-sm flex flex-col gap-6",
        className
      )}
    >
      {children}
    </motion.div>
  );
}
