import { motion } from 'framer-motion';

interface ProgressBarProps {
  value: number;
  max?: number;
  colorClass?: string;
  className?: string;
  height?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  labelFormat?: (val: number) => string;
}

export function ProgressBar({ 
  value, 
  max = 100, 
  colorClass = 'bg-primary', 
  className = '',
  height = 'md',
  showLabel = false,
  labelFormat = (v) => `${v.toFixed(1)}`
}: ProgressBarProps) {
  
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  
  const heightClass = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4'
  }[height];

  return (
    <div className={`w-full ${className}`}>
      {showLabel && (
        <div className="flex justify-end mb-1 text-xs font-mono opacity-80">
          {labelFormat(value)}
        </div>
      )}
      <div className={`w-full bg-background/50 rounded-full overflow-hidden border border-border ${heightClass}`}>
        <motion.div 
          className={`h-full rounded-full ${colorClass}`}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}
