import { Play, RotateCcw } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ToolActionsProps {
  onExecute?: () => void;
  onClear?: () => void;
  executeLabel?: string;
  clearLabel?: string;
  executeIcon?: LucideIcon;
  clearIcon?: LucideIcon;
  isExecuting?: boolean;
  executeDisabled?: boolean;
  className?: string;
}

export function ToolActions({
  onExecute,
  onClear,
  executeLabel = 'Execute',
  clearLabel = 'Clear',
  executeIcon: ExecuteIcon = Play,
  clearIcon: ClearIcon = RotateCcw,
  isExecuting = false,
  executeDisabled = false,
  className,
}: ToolActionsProps) {
  return (
    <div className={cn("flex items-center gap-3 pt-2", className)}>
      {onExecute && (
        <Button
          type="submit"
          onClick={(e) => {
            // Only call onExecute if it's not a form submit
            if (e.target instanceof HTMLButtonElement && e.target.type !== 'submit') {
              onExecute();
            }
          }}
          disabled={isExecuting || executeDisabled}
          className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
        >
          <ExecuteIcon className="w-4 h-4" />
          {isExecuting ? 'Executing...' : executeLabel}
        </Button>
      )}
      {onClear && (
        <Button
          type="button"
          variant="outline"
          onClick={onClear}
          className="gap-2"
          disabled={isExecuting}
        >
          <ClearIcon className="w-4 h-4" />
          {clearLabel}
        </Button>
      )}
    </div>
  );
}
