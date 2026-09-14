import { useState, useMemo } from 'react';
import { useLocation, NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  HelpCircle,
  ChevronDown,
  BookOpen,
  ArrowRight,
  Lightbulb,
} from 'lucide-react';
import { getSEOConfig, seoConfigMap } from '@/constants/seoConfig';
import { getToolKnowledge } from '@/constants/toolKnowledge';
import { blogArticles, type BlogArticle, type RelatedTool } from '@/content/blog/articles';
import { cn } from '@/lib/utils';

interface ToolSEOSectionProps {
  currentTitle?: string;
  className?: string;
}

export function ToolSEOSection({ currentTitle, className }: ToolSEOSectionProps) {
  const location = useLocation();
  const currentPath = location.pathname.length > 1 && location.pathname.endsWith('/')
    ? location.pathname.slice(0, -1)
    : location.pathname;

  const isExcluded =
    currentPath === '/' ||
    currentPath === '/settings' ||
    currentPath === '/404' ||
    currentPath === '/api-explorer';

  const seo = getSEOConfig(currentPath);

  // Retrieve customized How-To steps and FAQs for this specific tool or category
  const knowledge = useMemo(() => {
    if (isExcluded) return { howTo: [], faqs: [] };
    return getToolKnowledge(currentPath, seo.category);
  }, [currentPath, seo.category, isExcluded]);

  const howTo = knowledge.howTo;
  const faqs = (seo.faqs && seo.faqs.length > 0) ? seo.faqs : knowledge.faqs;

  // Find 3 related tools in the same category or overall suite
  const relatedTools = useMemo(() => {
    if (isExcluded) return [];
    const currentCategory = seo.category || '';
    const candidates: { path: string; name: string; description: string; category: string }[] = [];

    // First priority: same category
    for (const [r, config] of Object.entries(seoConfigMap)) {
      if (
        r !== currentPath &&
        r !== '/' &&
        r !== '/settings' &&
        r !== '/404' &&
        r !== '/api-explorer'
      ) {
        const isSameCategory = config.category && config.category === currentCategory;
        if (isSameCategory) {
          candidates.push({
            path: r,
            name: config.title.split(/[—|]/)[0].trim(),
            description: config.description,
            category: config.category || '',
          });
        }
      }
    }

    // If fewer than 3, add other security tools
    if (candidates.length < 3) {
      for (const [r, config] of Object.entries(seoConfigMap)) {
        if (
          r !== currentPath &&
          r !== '/' &&
          r !== '/settings' &&
          r !== '/404' &&
          !candidates.some((c) => c.path === r)
        ) {
          candidates.push({
            path: r,
            name: config.title.split(/[—|]/)[0].trim(),
            description: config.description,
            category: config.category || '',
          });
          if (candidates.length >= 3) break;
        }
      }
    }

    return candidates.slice(0, 3);
  }, [currentPath, seo, isExcluded]);

  // Find dedicated companion educational article if one exists for this tool
  const companionArticle = useMemo(() => {
    if (isExcluded) return null;
    return (
      blogArticles.find(
        (article) =>
          article.relatedTools.some((t) => t.path === currentPath) &&
          (article.slug.includes(currentPath.split('/').pop() || '') ||
            article.relatedTools[0]?.path === currentPath)
      ) || null
    );
  }, [currentPath, isExcluded]);

  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  const toggleFaq = (idx: number) => {
    setOpenFaqIndex((prev) => (prev === idx ? null : idx));
  };

  if (isExcluded) {
    return null;
  }

  const toolName = currentTitle || seo.title.split(/[—|]/)[0].trim();

  return (
    <section
      aria-label="Educational Guide and Frequently Asked Questions"
      className={cn('space-y-8 pt-8 border-t border-border mt-10', className)}
    >
      {/* Dedicated Companion Educational Guide Banner */}
      {companionArticle && (
        <div className="p-5 sm:p-6 rounded-2xl border border-primary/40 bg-gradient-to-r from-primary/10 via-secondary/60 to-background shadow-lg relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-48 h-48 bg-primary/10 rounded-full blur-3xl pointer-events-none group-hover:scale-125 transition-transform duration-500" />
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 relative z-10">
            <div className="space-y-1.5 max-w-2xl">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-primary/20 text-primary border border-primary/30">
                <BookOpen className="w-3.5 h-3.5" />
                <span>Deep-Dive Educational Guide</span>
              </div>
              <h3 className="text-lg sm:text-xl font-bold text-foreground group-hover:text-primary transition-colors">
                {companionArticle.title}
              </h3>
              <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed line-clamp-2">
                {companionArticle.description}
              </p>
            </div>
            <NavLink
              to={`/blog/${companionArticle.slug}`}
              className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-primary text-primary-foreground font-semibold text-xs sm:text-sm shadow-md hover:opacity-90 transition-all flex-shrink-0"
            >
              <span>Read Full Guide &amp; Proofs</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </NavLink>
          </div>
        </div>
      )}

      {/* 2-Column Guide & FAQs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Left: How-To & Technical Context */}
        <div className="rounded-xl border border-border bg-card/40 p-6 space-y-5">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-foreground tracking-tight">
                {knowledge.guideTitle || `How to Use ${toolName}`}
              </h2>
              <p className="text-xs text-muted-foreground">
                {knowledge.guideSubtitle || 'Step-by-step cryptographic workflow'}
              </p>
            </div>
          </div>

          <ol className="space-y-3.5 text-sm">
            {howTo.map((step) => (
              <li key={step.step} className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-secondary border border-border flex items-center justify-center text-xs font-semibold text-primary">
                  {step.step}
                </span>
                <div className="pt-0.5">
                  <strong className="text-foreground font-medium">{step.title}:</strong>
                  <p className="text-muted-foreground text-xs mt-0.5 leading-relaxed">
                    {step.description}
                  </p>
                </div>
              </li>
            ))}
          </ol>

          <div className="rounded-lg bg-secondary/40 border border-border/80 p-3.5 flex items-start gap-3">
            <Lightbulb className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-muted-foreground leading-relaxed">
              <span className="text-foreground font-medium">Pro-tip: </span>
              Keyboard shortcut <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-background border border-border rounded">Ctrl</kbd> + <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-background border border-border rounded">K</kbd> allows you to jump between any of the 40+ security tools instantly.
            </p>
          </div>
        </div>

        {/* Right: Interactive FAQ Accordion */}
        <div className="rounded-xl border border-border bg-card/40 p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
              <HelpCircle className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-foreground tracking-tight">
                Frequently Asked Questions
              </h2>
              <p className="text-xs text-muted-foreground">
                Verified answers to common queries
              </p>
            </div>
          </div>

          <div className="space-y-2.5 pt-1">
            {faqs.map((faq, idx) => {
              const isOpen = openFaqIndex === idx;
              return (
                <div
                  key={idx}
                  className="rounded-lg border border-border bg-card/60 overflow-hidden transition-colors hover:border-border/90"
                >
                  <button
                    type="button"
                    onClick={() => toggleFaq(idx)}
                    className="w-full flex items-center justify-between p-3.5 text-left text-sm font-medium text-foreground transition-colors hover:text-primary focus:outline-none"
                    aria-expanded={isOpen}
                  >
                    <span className="pr-3 leading-snug">{faq.question}</span>
                    <motion.div
                      animate={{ rotate: isOpen ? 180 : 0 }}
                      transition={{ duration: 0.2 }}
                      className="flex-shrink-0 text-muted-foreground"
                    >
                      <ChevronDown className="w-4 h-4" />
                    </motion.div>
                  </button>

                  <AnimatePresence initial={false}>
                    {isOpen && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.25, ease: 'easeInOut' }}
                      >
                        <div className="px-3.5 pb-3.5 pt-0 text-xs text-muted-foreground leading-relaxed border-t border-border/40 mt-1">
                          <p className="pt-2">{faq.answer}</p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Related Tools Internal Linking Graph */}
      {relatedTools.length > 0 && (
        <div className="space-y-4 pt-2">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-foreground tracking-tight flex items-center gap-2">
              <span>Related Tools in {seo.category || 'CipherVerse'}</span>
            </h2>
            <NavLink
              to="/classical"
              className="text-xs text-primary hover:underline flex items-center gap-1"
            >
              Explore all tools <ArrowRight className="w-3 h-3" />
            </NavLink>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {relatedTools.map((tool) => (
              <NavLink
                key={tool.path}
                to={tool.path}
                className="group p-4 rounded-xl border border-border bg-card/30 hover:bg-card/70 hover:border-primary/40 transition-all duration-200 flex flex-col justify-between"
              >
                <div>
                  <div className="text-[11px] font-medium text-primary mb-1">
                    {tool.category}
                  </div>
                  <h3 className="text-sm font-semibold text-foreground group-hover:text-primary transition-colors flex items-center justify-between">
                    <span>{tool.name}</span>
                    <ArrowRight className="w-3.5 h-3.5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all" />
                  </h3>
                  <p className="text-xs text-muted-foreground mt-1.5 line-clamp-2 leading-relaxed">
                    {tool.description}
                  </p>
                </div>
              </NavLink>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
