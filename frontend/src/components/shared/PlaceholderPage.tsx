import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { Construction } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface PlaceholderPageProps {
  title: string;
  description: string;
  icon: LucideIcon;
}

export function PlaceholderPage({ title, description, icon }: PlaceholderPageProps) {
  return (
    <ToolPageLayout title={title} description={description} icon={icon}>
      <div className="glass rounded-xl p-12 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-muted/50 flex items-center justify-center">
          <Construction className="w-8 h-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold text-foreground mb-2">Coming Soon</h3>
        <p className="text-sm text-muted-foreground max-w-md mx-auto">
          This module is under development and will be available in a future update.
        </p>
      </div>
    </ToolPageLayout>
  );
}
