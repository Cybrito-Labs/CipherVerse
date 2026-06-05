import { LucideIcon } from 'lucide-react';

interface RiskCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon?: LucideIcon;
  riskLevel?: 'low' | 'medium' | 'high' | 'neutral';
  className?: string;
}

export function RiskCard({ title, value, description, icon: Icon, riskLevel = 'neutral', className = '' }: RiskCardProps) {
  const getColors = () => {
    switch (riskLevel) {
      case 'low': return 'border-success/30 bg-success/10 text-success';
      case 'medium': return 'border-warning/30 bg-warning/10 text-warning';
      case 'high': return 'border-destructive/30 bg-destructive/10 text-destructive';
      default: return 'border-border bg-background/50 text-foreground';
    }
  };

  const getIconColor = () => {
    switch (riskLevel) {
      case 'low': return 'text-success';
      case 'medium': return 'text-warning';
      case 'high': return 'text-destructive';
      default: return 'text-primary';
    }
  };

  return (
    <div className={`p-6 rounded-xl border ${getColors()} ${className} flex flex-col justify-between`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold tracking-tight text-sm uppercase opacity-80">{title}</h3>
        {Icon && <Icon className={`w-5 h-5 opacity-80 ${getIconColor()}`} />}
      </div>
      <div>
        <div className="text-3xl font-black tracking-tighter truncate" title={value.toString()}>
          {value}
        </div>
        {description && (
          <p className="text-xs mt-2 opacity-70 leading-snug">{description}</p>
        )}
      </div>
    </div>
  );
}
