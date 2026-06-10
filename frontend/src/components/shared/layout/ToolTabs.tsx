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
    <div className={cn("bg-[#000000] p-1 rounded-lg border border-[#27272A] flex w-full max-w-[300px]", className)}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onTabChange(tab.id)}
          className={cn(
            "flex-1 px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200",
            activeTab === tab.id
              ? "bg-[#27272A] text-[#EDEDED] shadow-sm"
              : "text-[#A1A1AA] hover:text-[#EDEDED]"
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
