import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import type { LucideIcon } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { ArrowRight } from 'lucide-react';

interface CategoryCardProps {
  title: string;
  description: string;
  icon: LucideIcon;
  path: string;
  toolCount?: number;
  className?: string;
  delay?: number;
}

export function CategoryCard({
  title,
  description,
  icon: Icon,
  path,
  toolCount,
  className,
  delay = 0,
}: CategoryCardProps) {
  const navigate = useNavigate();

  return (
    <motion.button
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      whileHover={{ y: -4, scale: 1.01 }}
      whileTap={{ scale: 0.98 }}
      onClick={() => navigate(path)}
      className={cn(
        'group w-full text-left p-6 rounded-2xl',
        'glass glow-cyan-hover',
        'transition-all duration-300',
        'hover:border-primary/30',
        className
      )}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
          <Icon className="w-6 h-6 text-primary" />
        </div>
        {toolCount !== undefined && (
          <Badge variant="secondary" className="bg-secondary/50">
            {toolCount} tools
          </Badge>
        )}
      </div>
      <h3 className="text-lg font-semibold text-foreground mb-1.5 group-hover:text-primary transition-colors">
        {title}
      </h3>
      <p className="text-sm text-muted-foreground line-clamp-2">{description}</p>
      <div className="mt-4 flex items-center text-sm text-primary opacity-0 group-hover:opacity-100 transition-opacity">
        Explore
        <ArrowRight className="w-4 h-4 ml-1.5 group-hover:translate-x-1 transition-transform" />
      </div>
    </motion.button>
  );
}
