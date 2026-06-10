import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Settings2 } from 'lucide-react';

interface ToolResultPanelProps {
  title?: string;
  result?: string | null;
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  onClear?: () => void;
  className?: string;
  children?: ReactNode; // For custom results
  emptyMessage?: string; // e.g. "Enter your data to see results"
}

export function ToolResultPanel({
  title = 'Result',
  result,
  isLoading,
  error,
  onRetry,
  onClear,
  className,
  children,
  emptyMessage = "Enter your data and execute to see results appear here",
}: ToolResultPanelProps) {
  const hasData = result !== undefined && result !== null;
  const hasChildren = children !== undefined && children !== null;
  const isPending = isLoading === true;
  const hasError = error !== null && error !== undefined;
  const isEmpty = !hasData && !hasChildren && !isPending && !hasError;

  return (
    <motion.div
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
      className={cn("sticky top-24", className)}
    >
      <ResultPanel
        title={title}
        result={result}
        isLoading={isLoading}
        error={error}
        onRetry={onRetry}
        onClear={onClear}
      >
        {children}
      </ResultPanel>
      
      {isEmpty && (
        <div className="rounded-[14px] border border-border border-dashed bg-background p-12 text-center flex flex-col items-center justify-center gap-3">
          <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center border border-border">
            <Settings2 className="w-5 h-5 text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground max-w-[250px]">
            {emptyMessage}
          </p>
        </div>
      )}
    </motion.div>
  );
}
