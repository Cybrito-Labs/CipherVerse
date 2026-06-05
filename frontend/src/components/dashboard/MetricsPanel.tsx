import { ReactNode } from 'react';

interface MetricsPanelProps {
  title: string;
  children: ReactNode;
  className?: string;
}

export function MetricsPanel({ title, children, className = '' }: MetricsPanelProps) {
  return (
    <div className={`glass rounded-xl overflow-hidden ${className}`}>
      <div className="bg-background/40 border-b border-border p-4">
        <h3 className="font-semibold text-primary uppercase tracking-wider text-sm">{title}</h3>
      </div>
      <div className="p-0">
        <dl className="divide-y divide-border m-0">
          {children}
        </dl>
      </div>
    </div>
  );
}

interface MetricRowProps {
  label: string;
  value: string | number | ReactNode;
  isMono?: boolean;
}

export function MetricRow({ label, value, isMono = false }: MetricRowProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 px-4 py-3 bg-background/20 hover:bg-background/40 transition-colors">
      <dt className="text-xs font-medium text-muted-foreground flex items-center sm:col-span-1">
        {label}
      </dt>
      <dd className={`text-sm sm:col-span-2 break-all ${isMono ? 'font-mono' : ''}`}>
        {value}
      </dd>
    </div>
  );
}
