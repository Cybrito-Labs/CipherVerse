import { Link } from 'react-router-dom';
import { type RelatedTool } from '@/content/blog/articles';
import { ArrowRight, Sparkles, Wrench } from 'lucide-react';

interface ToolEmbedCTAProps {
  tool: RelatedTool;
  titleOverride?: string;
}

export function ToolEmbedCTA({ tool, titleOverride }: ToolEmbedCTAProps) {
  return (
    <div className="my-8 p-5 sm:p-6 rounded-xl border border-primary/20 bg-gradient-to-r from-primary/5 via-secondary/40 to-background shadow-md relative overflow-hidden group">
      <div className="absolute top-0 right-0 w-32 h-32 bg-primary/10 rounded-full blur-3xl pointer-events-none group-hover:scale-150 transition-transform duration-500" />

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 relative z-10">
        <div className="space-y-1 max-w-xl">
          <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-primary/10 text-primary border border-primary/20">
            <Sparkles className="w-3 h-3" />
            <span>Interactive Tool Workbench</span>
          </div>
          <h4 className="text-base sm:text-lg font-semibold text-foreground group-hover:text-primary transition-colors flex items-center gap-2">
            <Wrench className="w-4 h-4 text-primary" />
            {titleOverride || tool.name}
          </h4>
          <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
            {tool.description}
          </p>
        </div>

        <Link
          to={tool.path}
          className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-primary text-primary-foreground font-medium text-xs sm:text-sm shadow-sm hover:opacity-90 transition-all flex-shrink-0"
        >
          <span>Launch Tool</span>
          <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
        </Link>
      </div>
    </div>
  );
}
