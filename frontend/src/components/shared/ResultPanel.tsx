import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { CopyButton } from './CopyButton';
import { DownloadButton } from './DownloadButton';
import { LoadingState } from './LoadingState';
import { ErrorState } from './ErrorState';
import { Trash2, CheckCircle2 } from 'lucide-react';

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
  onClear,
  className,
  children,
}: ResultPanelProps) {
  if (isLoading) {
    return (
      <div className={cn('rounded-[14px] border border-[#27272A] bg-[#0A0A0A] p-6 shadow-sm', className)}>
        <LoadingState />
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('rounded-[14px] border border-destructive/30 bg-[#7F1D1D]/10 p-6', className)}>
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
      className={cn('rounded-[14px] border border-[#27272A] bg-[#000000] p-5 shadow-sm flex flex-col', className)}
    >
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-[#27272A]">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-[#14532D]" />
          <h3 className="text-[13px] font-semibold text-[#EDEDED] uppercase tracking-wider">
            {title}
          </h3>
        </div>
        <div className="flex items-center gap-2">
          {result && <CopyButton text={result} variant="button" />}
          {result && <DownloadButton content={result} filename="cipherverse-output.txt" />}
          {onClear && (
            <button
              onClick={onClear}
              className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors hover:bg-destructive/10 hover:text-destructive h-9 px-3 text-[#A1A1AA]"
              title="Clear Output"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
      {result && (
        <div className="relative group flex-1">
          <pre className="p-4 rounded-xl bg-[#0A0A0A] border border-[#27272A]/50 text-[13px] font-mono text-[#EDEDED] whitespace-pre-wrap break-all overflow-auto max-h-[500px] shadow-inner selection:bg-indigo-500/30">
            {result}
          </pre>
        </div>
      )}
      {children && <div className="mt-4">{children}</div>}
    </motion.div>
  );
}
