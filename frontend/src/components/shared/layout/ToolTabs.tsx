import { cn } from '@/lib/utils';

interface ToolTabItem {
  id: string;
  label: string;
}

interface ToolTabsProps {
  tabs: ToolTabItem[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  className?: string;
}

export function ToolTabs({ tabs, activeTab, onTabChange, className }: ToolTabsProps) {
  if (!tabs || tabs.length === 0) return null;

  return (
    <div className={cn("bg-background p-1 rounded-lg border border-border flex w-full overflow-x-auto whitespace-nowrap scrollbar-none", className)}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onTabChange(tab.id)}
          className={cn(
            "flex-1 shrink-0 px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200",
            activeTab === tab.id
              ? "bg-input text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
