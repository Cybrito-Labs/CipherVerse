import { ChevronUp, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface RotorControlProps {
  id: string;
  label: string;
  value: number; // 0-25
  onChange: (newValue: number) => void;
  max?: number;
  showLetters?: boolean;
}

export function RotorControl({ label, value, onChange, max = 25, showLetters = true }: RotorControlProps) {
  const handleIncrement = () => {
    onChange(value >= max ? 0 : value + 1);
  };

  const handleDecrement = () => {
    onChange(value <= 0 ? max : value - 1);
  };

  const displayValue = showLetters 
    ? String.fromCharCode(65 + value) // 0=A, 1=B
    : value.toString().padStart(2, '0');

  return (
    <div className="flex flex-col items-center bg-black/40 p-3 rounded-lg border border-border/50 shadow-inner w-24">
      <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-2 opacity-70">
        {label}
      </span>
      
      <button 
        type="button"
        onClick={handleIncrement}
        className="w-full flex justify-center py-1 text-muted-foreground hover:text-primary hover:bg-white/5 rounded transition-colors"
      >
        <ChevronUp className="w-5 h-5" />
      </button>

      <div className="relative w-16 h-16 bg-gradient-to-b from-neutral-800 to-neutral-950 rounded-md border-y-2 border-border/80 shadow-[inset_0_4px_10px_rgba(0,0,0,0.5)] overflow-hidden my-1 flex items-center justify-center">
        {/* Simulating the mechanical window */}
        <div className="absolute inset-0 bg-[linear-gradient(to_bottom,rgba(0,0,0,0.8)_0%,transparent_30%,transparent_70%,rgba(0,0,0,0.8)_100%)] pointer-events-none z-10"></div>
        
        <AnimatePresence mode="popLayout">
          <motion.span
            key={value}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -20, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="text-3xl font-black font-mono text-white tracking-tighter"
          >
            {displayValue}
          </motion.span>
        </AnimatePresence>
      </div>

      <button 
        type="button"
        onClick={handleDecrement}
        className="w-full flex justify-center py-1 text-muted-foreground hover:text-primary hover:bg-white/5 rounded transition-colors"
      >
        <ChevronDown className="w-5 h-5" />
      </button>
    </div>
  );
}
