import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { CopyButton } from './CopyButton';
import { LoadingState } from './LoadingState';
import { ErrorState } from './ErrorState';

interface ResultPanelProps {
  title?: string;
  result?: string | null;
  isLoading?: boolean;
  error?: Error | null;
  onRetry?: () => void;
  onClear?: () => void;
  className?: string;
  children?: ReactNode;
}

export function ResultPanel({
  title = 'Output',
  result,
  isLoading,
  error,
  onRetry,
  className,
  children,
}: ResultPanelProps) {
  if (isLoading) {
    return (
      <div className={cn('rounded-xl border border-[#27272A] bg-[#0A0A0A] p-6', className)}>
        <LoadingState />
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('rounded-xl border border-destructive/20 bg-destructive/5 p-6', className)}>
        <ErrorState message={error.message} onRetry={onRetry} />
      </div>
    );
  }

  if (!result && !children) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className={cn('rounded-xl border border-[#27272A] bg-[#0A0A0A] p-6 space-y-4 shadow-sm', className)}
    >
      <div className="flex items-center justify-between pb-2 border-b border-[#27272A]/50">
        <h3 className="text-sm font-medium text-[#A1A1AA] tracking-tight">
          {title}
        </h3>
        {result && <CopyButton text={result} variant="button" />}
      </div>
      {result && (
        <pre className="p-4 rounded-lg bg-black border border-[#27272A] text-sm font-mono text-[#EDEDED] whitespace-pre-wrap break-all overflow-auto max-h-[400px]">
          {result}
        </pre>
      )}
      {children}
    </motion.div>
  );
}
