import { Link } from 'react-router-dom';
import { type BlogArticle } from '@/content/blog/articles';
import { Clock, Calendar, ArrowRight, User, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

interface BlogCardProps {
  article: BlogArticle;
  featured?: boolean;
}

export function BlogCard({ article, featured = false }: BlogCardProps) {
  return (
    <article
      className={cn(
        'group rounded-xl border border-border bg-card/70 backdrop-blur-md overflow-hidden transition-all duration-300 hover:border-primary/40 hover:shadow-lg hover:shadow-primary/5 flex flex-col',
        featured ? 'md:grid md:grid-cols-12 md:gap-6 border-primary/30 bg-card/90' : ''
      )}
    >
      {/* Visual Accent Banner */}
      <div
        className={cn(
          'relative overflow-hidden bg-gradient-to-br flex items-center justify-center p-6',
          article.coverGradient,
          featured ? 'md:col-span-5 min-h-[220px]' : 'h-40'
        )}
      >
        <div className="absolute inset-0 bg-background/20 backdrop-blur-[2px]" />
        <span className="relative z-10 font-mono text-3xl font-extrabold tracking-widest text-foreground/40 select-none group-hover:scale-105 group-hover:text-foreground/60 transition-all duration-300">
          [//CIPHER]
        </span>
        <div className="absolute bottom-3 left-3 z-10">
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-background/80 text-foreground border border-border backdrop-blur-md">
            {article.category}
          </span>
        </div>
      </div>

      {/* Content Body */}
      <div className={cn('p-5 sm:p-6 flex flex-col justify-between flex-1', featured ? 'md:col-span-7' : '')}>
        <div className="space-y-2.5">
          {/* Metadata Row */}
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <Calendar className="w-3.5 h-3.5" />
              <time dateTime={article.publishedAt}>
                {new Date(article.publishedAt).toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                })}
              </time>
            </span>
            <span>•</span>
            <span className="inline-flex items-center gap-1">
              <Clock className="w-3.5 h-3.5" />
              <span>{article.readTime}</span>
            </span>
          </div>

          {article.seriesBadge && (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30">
              <Sparkles className="w-3 h-3 text-amber-400" />
              <span>{article.seriesBadge}</span>
            </div>
          )}

          {/* Title */}
          <h3
            className={cn(
              'font-semibold text-foreground group-hover:text-primary transition-colors tracking-tight',
              featured ? 'text-xl sm:text-2xl leading-tight' : 'text-lg leading-snug line-clamp-2'
            )}
          >
            <Link to={`/blog/${article.slug}`} className="focus:outline-none">
              <span className="absolute inset-0 z-10 md:hidden" aria-hidden="true" />
              {article.title}
            </Link>
          </h3>

          {/* Description / Excerpt */}
          <p className="text-xs sm:text-sm text-muted-foreground line-clamp-3 leading-relaxed">
            {article.description}
          </p>

          {/* Tags */}
          <div className="flex flex-wrap gap-1.5 pt-1">
            {article.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="text-[11px] px-2 py-0.5 rounded-md bg-secondary/80 text-secondary-foreground font-mono"
              >
                #{tag}
              </span>
            ))}
          </div>
        </div>

        {/* Footer row */}
        <div className="pt-4 mt-4 border-t border-border/60 flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <User className="w-3.5 h-3.5 text-primary" />
            <span className="truncate max-w-[160px]">{article.author.name}</span>
          </div>

          <Link
            to={`/blog/${article.slug}`}
            className="inline-flex items-center gap-1 text-xs font-semibold text-primary hover:underline"
          >
            <span>Read Article</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>
      </div>
    </article>
  );
}
