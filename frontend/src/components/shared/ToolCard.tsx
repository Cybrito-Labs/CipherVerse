import { useNavigate } from 'react-router-dom';
import { motion, useMotionTemplate, useMotionValue } from 'framer-motion';
import { cn } from '@/lib/utils';
import type { LucideIcon } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { ArrowRight } from 'lucide-react';
import { MouseEvent } from 'react';

interface ToolCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
  path: string;
  badge?: string;
  className?: string;
}

export function ToolCard({ title, description, icon: Icon, path, badge, className }: ToolCardProps) {
  const navigate = useNavigate();
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  function handleMouseMove({ currentTarget, clientX, clientY }: MouseEvent) {
    const { left, top } = currentTarget.getBoundingClientRect();
    mouseX.set(clientX - left);
    mouseY.set(clientY - top);
  }

  return (
    <motion.button
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.98 }}
      onClick={() => navigate(path)}
      onMouseMove={handleMouseMove}
      className={cn(
        'group relative w-full text-left p-6 rounded-xl',
        'bg-[#0A0A0A] border border-[#27272A]',
        'transition-all duration-300',
        'hover:border-[#52525B] hover:shadow-[0_8px_30px_rgb(0,0,0,0.12)]',
        'overflow-hidden',
        className
      )}
    >
      <motion.div
        className="pointer-events-none absolute -inset-px rounded-xl opacity-0 transition duration-300 group-hover:opacity-100"
        style={{
          background: useMotionTemplate`
            radial-gradient(
              300px circle at ${mouseX}px ${mouseY}px,
              rgba(255, 255, 255, 0.05),
              transparent 80%
            )
          `,
        }}
      />
      
      <div className="relative z-10 flex items-start justify-between mb-4">
        <div className="w-10 h-10 rounded-lg bg-[#171717] border border-[#27272A] flex items-center justify-center group-hover:bg-[#27272A] transition-colors duration-300">
          <Icon className="w-5 h-5 text-[#EDEDED]" />
        </div>
        {badge && (
          <Badge variant="secondary" className="text-[10px] bg-[#171717] border border-[#27272A]">
            {badge}
          </Badge>
        )}
      </div>
      
      <h3 className="relative z-10 font-semibold text-[#EDEDED] mb-2 group-hover:text-white transition-colors duration-300 tracking-tight">
        {title}
      </h3>
      
      <p className="relative z-10 text-xs text-[#A1A1AA] line-clamp-2 leading-relaxed">
        {description}
      </p>
      
      <div className="relative z-10 mt-4 flex items-center text-xs font-medium text-[#EDEDED] opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-300">
        Open tool
        <ArrowRight className="w-3 h-3 ml-1" />
      </div>
    </motion.button>
  );
}
