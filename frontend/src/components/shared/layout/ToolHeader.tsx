import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import type { LucideIcon } from 'lucide-react';

interface ToolHeaderProps {
  title: string;
  description: string;
  icon: LucideIcon;
  badges?: { label: string; variant?: 'default' | 'success' | 'warning' }[];
  className?: string;
}

export function ToolHeader({
  title,
  description,
  icon: Icon,
  badges,
  className,
}: ToolHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className={cn("flex items-start gap-5 border-b border-border pb-6", className)}
    >
      <div className="w-12 h-12 rounded-lg bg-card border border-border flex items-center justify-center flex-shrink-0 shadow-sm mt-0.5">
        <Icon className="w-5 h-5 text-foreground" />
      </div>
      <div className="pt-0.5 flex-1">
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-semibold text-foreground tracking-tight">{title}</h1>
          {badges && badges.length > 0 && (
            <div className="flex gap-2">
              {badges.map((badge, idx) => (
                <span
                  key={idx}
                  className={cn(
                    "px-2 py-0.5 text-xs font-medium rounded-md border",
                    badge.variant === 'success' ? "bg-[#14532D]/30 border-[#14532D] text-[#F0FDF4]" :
                    badge.variant === 'warning' ? "bg-[#78350F]/30 border-[#78350F] text-[#FFFBEB]" :
                    "bg-secondary border-border text-muted-foreground"
                  )}
                >
                  {badge.label}
                </span>
              ))}
            </div>
          )}
        </div>
        <p className="text-[15px] text-muted-foreground mt-2 max-w-3xl leading-relaxed">{description}</p>
      </div>
    </motion.div>
  );
}
