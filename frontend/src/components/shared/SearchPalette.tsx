import { useState, useMemo, useEffect } from 'react';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Search, CornerDownLeft, ArrowUp, ArrowDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { allNavItems } from '@/constants/navigation';
import { toolDefinitions } from '@/constants/tools';
import { ScrollArea } from '@/components/ui/scroll-area';

interface SearchPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate: (path: string) => void;
}

interface SearchResult {
  label: string;
  description: string;
  path: string;
  category: string;
}

export function SearchPalette({ open, onOpenChange, onNavigate }: SearchPaletteProps) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);

  const allItems = useMemo<SearchResult[]>(() => {
    const items: SearchResult[] = [];
    allNavItems.forEach((item) => items.push({ label: item.label, description: item.description, path: item.path, category: 'Pages' }));
    toolDefinitions.forEach((tool) => items.push({ label: tool.name, description: tool.description, path: tool.path, category: tool.category }));
    return items;
  }, []);

  const results = useMemo(() => {
    if (!query.trim()) return allItems.slice(0, 12);
    const q = query.toLowerCase();
    return allItems.filter(item => 
      item.label.toLowerCase().includes(q) || 
      item.description.toLowerCase().includes(q) || 
      item.category.toLowerCase().includes(q)
    );
  }, [query, allItems]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSelectedIndex(0);
  }, [query, open]);

  // Handle keyboard navigation inside the palette
  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev < results.length - 1 ? prev + 1 : prev));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev > 0 ? prev - 1 : prev));
      } else if (e.key === 'Enter' && results[selectedIndex]) {
        e.preventDefault();
        onNavigate(results[selectedIndex].path);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, results, selectedIndex, onNavigate]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl p-0 gap-0 bg-background/80 backdrop-blur-2xl border border-border shadow-2xl overflow-hidden rounded-xl">
        <div className="flex items-center px-4 border-b border-border/50 bg-card/50">
          <Search className="w-4 h-4 text-muted-foreground flex-shrink-0" />
          <Input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search tools, algorithms, or pages..."
            className="border-0 focus-visible:ring-0 bg-transparent text-lg h-14 text-foreground placeholder:text-muted-foreground"
          />
        </div>
        <ScrollArea className="max-h-[380px] bg-background/40">
          <div className="p-2">
            {results.map((item, index) => {
              const isSelected = index === selectedIndex;
              return (
                <button
                  key={item.path + item.label}
                  onClick={() => onNavigate(item.path)}
                  onMouseEnter={() => setSelectedIndex(index)}
                  className={cn(
                    'w-full flex items-center justify-between gap-3 px-3 py-3 rounded-lg',
                    'text-left transition-colors duration-150 group',
                    isSelected ? 'bg-secondary text-foreground' : 'hover:bg-card text-muted-foreground'
                  )}
                >
                  <div className="flex flex-col min-w-0">
                    <p className={cn("font-medium truncate transition-colors", isSelected ? 'text-foreground' : 'text-muted-foreground group-hover:text-foreground')}>
                      {item.label}
                    </p>
                    <p className={cn("text-xs truncate transition-colors mt-0.5", isSelected ? 'text-muted-foreground' : 'text-muted-foreground')}>
                      {item.category} <span className="opacity-50 mx-1">•</span> {item.description}
                    </p>
                  </div>
                  {isSelected && (
                    <CornerDownLeft className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                  )}
                </button>
              );
            })}
            {results.length === 0 && (
              <div className="py-12 text-center text-sm text-muted-foreground">
                No results found for "{query}"
              </div>
            )}
          </div>
        </ScrollArea>
        {/* Footer hints */}
        <div className="border-t border-border/50 bg-card/80 px-4 py-2 flex items-center justify-end gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1"><ArrowUp className="w-3 h-3"/><ArrowDown className="w-3 h-3"/> Navigate</span>
          <span className="flex items-center gap-1"><CornerDownLeft className="w-3 h-3"/> Select</span>
          <span className="flex items-center gap-1"><kbd className="font-sans px-1 py-0.5 bg-secondary border border-border rounded">ESC</kbd> Close</span>
        </div>
      </DialogContent>
    </Dialog>
  );
}
