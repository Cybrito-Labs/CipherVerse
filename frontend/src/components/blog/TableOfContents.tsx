import { useEffect, useState } from 'react';
import { type TableOfContentsItem } from '@/content/blog/articles';
import { cn } from '@/lib/utils';
import { ListCollapse } from 'lucide-react';

interface TableOfContentsProps {
  items: TableOfContentsItem[];
}

export function TableOfContents({ items }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>(items[0]?.id || '');

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visibleEntry = entries.find((e) => e.isIntersecting);
        if (visibleEntry) {
          setActiveId(visibleEntry.target.id);
        }
      },
      {
        rootMargin: '-80px 0px -60% 0px',
        threshold: 0.1,
      }
    );

    items.forEach((item) => {
      const el = document.getElementById(item.id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [items]);

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      const yOffset = -90;
      const y = el.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: y, behavior: 'smooth' });
    }
  };

  return (
    <nav aria-label="Table of contents" className="p-4 rounded-xl border border-border bg-card/60 backdrop-blur-sm">
      <div className="flex items-center gap-2 pb-3 mb-3 border-b border-border text-foreground font-semibold text-sm">
        <ListCollapse className="w-4 h-4 text-primary" />
        <span>Table of Contents</span>
      </div>
      <ul className="space-y-1.5 text-xs">
        {items.map((item) => {
          const isActive = activeId === item.id;
          return (
            <li key={item.id} style={{ paddingLeft: item.level > 2 ? `${(item.level - 2) * 12}px` : 0 }}>
              <button
                type="button"
                onClick={() => scrollTo(item.id)}
                className={cn(
                  'w-full text-left py-1 px-2 rounded-md transition-all duration-200 line-clamp-2 block',
                  isActive
                    ? 'bg-primary/10 text-primary font-medium border-l-2 border-primary'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'
                )}
              >
                {item.title}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
