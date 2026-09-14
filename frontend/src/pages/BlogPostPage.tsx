import { useParams, Link } from 'react-router-dom';
import { blogArticles, type BlogArticle } from '@/content/blog/articles';
import { TableOfContents } from '@/components/blog/TableOfContents';
import { ToolEmbedCTA } from '@/components/blog/ToolEmbedCTA';
import {
  Calendar,
  Clock,
  User,
  ArrowLeft,
  ArrowRight,
  Share2,
  Copy,
  Check,
  Info,
  AlertTriangle,
  Lightbulb,
  BookOpen,
  Sparkles,
} from 'lucide-react';
import { useState, useMemo } from 'react';
import { toast } from 'sonner';

export default function BlogPostPage() {
  const { slug } = useParams<{ slug: string }>();
  const [copiedCodeIdx, setCopiedCodeIdx] = useState<number | null>(null);
  const [copiedLink, setCopiedLink] = useState(false);

  const articleIndex = useMemo(() => {
    return blogArticles.findIndex((a) => a.slug === slug);
  }, [slug]);

  const article: BlogArticle | undefined = blogArticles[articleIndex];
  const prevArticle = articleIndex > 0 ? blogArticles[articleIndex - 1] : null;
  const nextArticle =
    articleIndex >= 0 && articleIndex < blogArticles.length - 1
      ? blogArticles[articleIndex + 1]
      : null;

  if (!article) {
    return (
      <div className="min-h-[70vh] flex flex-col items-center justify-center p-6 text-center space-y-4">
        <BookOpen className="w-12 h-12 text-muted-foreground" />
        <h1 className="text-2xl font-bold text-foreground">Article Not Found</h1>
        <p className="text-sm text-muted-foreground max-w-md">
          The cryptography article you are looking for does not exist or may have been moved.
        </p>
        <Link
          to="/blog"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition-opacity"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Knowledge Hub</span>
        </Link>
      </div>
    );
  }

  const copyCode = (code: string, idx: number) => {
    navigator.clipboard.writeText(code);
    setCopiedCodeIdx(idx);
    toast.success('Code copied to clipboard');
    setTimeout(() => setCopiedCodeIdx(null), 2000);
  };

  const copyArticleLink = () => {
    navigator.clipboard.writeText(window.location.href);
    setCopiedLink(true);
    toast.success('Article link copied to clipboard');
    setTimeout(() => setCopiedLink(false), 2000);
  };

  const shareOnTwitter = () => {
    const url = encodeURIComponent(window.location.href);
    const text = encodeURIComponent(`Read "${article.title}" on CipherVerse:`);
    window.open(`https://twitter.com/intent/tweet?text=${text}&url=${url}`, '_blank');
  };

  const shareOnLinkedIn = () => {
    const url = encodeURIComponent(window.location.href);
    window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${url}`, '_blank');
  };

  return (
    <article className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto">

      {/* Back Link & Breadcrumbs */}
      <div className="mb-6 flex items-center justify-between text-xs text-muted-foreground">
        <Link
          to="/blog"
          className="inline-flex items-center gap-1.5 hover:text-foreground transition-colors font-medium"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to All Articles</span>
        </Link>
        <span className="hidden sm:inline font-mono">
          {article.category}
        </span>
      </div>

      {/* Article Header */}
      <header className="space-y-4 pb-8 border-b border-border">
        <div className="flex flex-wrap items-center gap-2">
          {article.seriesBadge && (
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>{article.seriesBadge}</span>
            </div>
          )}
          <div className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20">
            {article.category}
          </div>
        </div>

        <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-foreground tracking-tight leading-tight">
          {article.title}
        </h1>

        <p className="text-sm sm:text-base text-muted-foreground leading-relaxed max-w-3xl">
          {article.description}
        </p>

        {/* Metadata & Author Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center text-primary font-bold text-xs">
              <User className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs sm:text-sm font-semibold text-foreground">
                {article.author.name}
              </div>
              <div className="text-[11px] text-muted-foreground">
                {article.author.role}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <Calendar className="w-3.5 h-3.5" />
              <time dateTime={article.publishedAt}>
                {new Date(article.publishedAt).toLocaleDateString('en-US', {
                  month: 'long',
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
        </div>
      </header>

      {/* Main Content Layout with Sticky Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 pt-8">
        {/* Left / Center Column: Article Content */}
        <div className="lg:col-span-8 space-y-10 min-w-0">
          {article.sections.map((section, sIdx) => (
            <section key={section.id} id={section.id} className="space-y-4 scroll-mt-24">
              <h2 className="text-xl sm:text-2xl font-bold text-foreground tracking-tight border-b border-border/40 pb-2">
                {section.heading}
              </h2>

              {section.paragraphs?.map((p, pIdx) => (
                <p key={pIdx} className="text-sm sm:text-base text-foreground/90 leading-relaxed">
                  {p}
                </p>
              ))}

              {/* Callout Box */}
              {section.callout && (
                <div
                  className={`p-4 rounded-xl border flex items-start gap-3 my-4 ${
                    section.callout.type === 'warning'
                      ? 'bg-amber-500/10 border-amber-500/30 text-amber-200'
                      : section.callout.type === 'tip'
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200'
                      : 'bg-primary/10 border-primary/30 text-foreground'
                  }`}
                >
                  <div className="mt-0.5 flex-shrink-0">
                    {section.callout.type === 'warning' ? (
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                    ) : section.callout.type === 'tip' ? (
                      <Lightbulb className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <Info className="w-4 h-4 text-primary" />
                    )}
                  </div>
                  <div className="space-y-1 text-xs sm:text-sm">
                    <div className="font-semibold text-foreground">
                      {section.callout.title}
                    </div>
                    <div className="text-muted-foreground leading-relaxed">
                      {section.callout.text}
                    </div>
                  </div>
                </div>
              )}

              {/* Code Snippet Block */}
              {section.codeBlock && (
                <div className="my-5 rounded-xl border border-border bg-slate-950 overflow-hidden shadow-md">
                  <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800 bg-slate-900/80 text-xs font-mono text-slate-400">
                    <span>{section.codeBlock.caption || section.codeBlock.language}</span>
                    <button
                      type="button"
                      onClick={() => copyCode(section.codeBlock!.code, sIdx)}
                      className="inline-flex items-center gap-1 hover:text-slate-200 transition-colors"
                    >
                      {copiedCodeIdx === sIdx ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-emerald-400">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <pre className="p-4 text-xs sm:text-sm font-mono text-slate-100 overflow-x-auto leading-relaxed">
                    <code>{section.codeBlock.code}</code>
                  </pre>
                </div>
              )}

              {section.postCodeParagraphs?.map((p, pIdx) => (
                <p key={`post-${pIdx}`} className="text-sm sm:text-base text-foreground/90 leading-relaxed">
                  {p}
                </p>
              ))}

              {/* Inline Tool Callout */}
              {section.toolCta && <ToolEmbedCTA tool={section.toolCta} />}
            </section>
          ))}

          {/* Share & Social Row */}
          <div className="pt-8 border-t border-border flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5">
                <Share2 className="w-3.5 h-3.5 text-primary" />
                <span>Share this guide:</span>
              </span>
              <button
                onClick={shareOnTwitter}
                className="px-2.5 py-1 rounded-md text-xs font-medium border border-border bg-card hover:bg-secondary transition-colors"
              >
                Twitter / X
              </button>
              <button
                onClick={shareOnLinkedIn}
                className="px-2.5 py-1 rounded-md text-xs font-medium border border-border bg-card hover:bg-secondary transition-colors"
              >
                LinkedIn
              </button>
              <button
                onClick={copyArticleLink}
                className="px-2.5 py-1 rounded-md text-xs font-medium border border-border bg-card hover:bg-secondary transition-colors inline-flex items-center gap-1"
              >
                {copiedLink ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copiedLink ? 'Copied' : 'Copy Link'}</span>
              </button>
            </div>

            {/* Tags */}
            <div className="flex flex-wrap gap-1.5">
              {article.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-[11px] px-2 py-0.5 rounded-md bg-secondary text-secondary-foreground font-mono"
                >
                  #{tag}
                </span>
              ))}
            </div>
          </div>

          {/* Next / Previous Article Navigation */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-6 border-t border-border">
            {prevArticle ? (
              <Link
                to={`/blog/${prevArticle.slug}`}
                className="p-4 rounded-xl border border-border bg-card hover:border-primary/40 transition-all text-left group space-y-1 block"
              >
                <div className="text-[11px] font-medium text-muted-foreground flex items-center gap-1">
                  <ArrowLeft className="w-3 h-3 group-hover:-translate-x-1 transition-transform" />
                  <span>Previous Article</span>
                </div>
                <div className="text-sm font-semibold text-foreground group-hover:text-primary transition-colors line-clamp-1">
                  {prevArticle.title}
                </div>
              </Link>
            ) : (
              <div />
            )}

            {nextArticle ? (
              <Link
                to={`/blog/${nextArticle.slug}`}
                className="p-4 rounded-xl border border-border bg-card hover:border-primary/40 transition-all text-right group space-y-1 block"
              >
                <div className="text-[11px] font-medium text-muted-foreground flex items-center justify-end gap-1">
                  <span>Next Article</span>
                  <ArrowRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                </div>
                <div className="text-sm font-semibold text-foreground group-hover:text-primary transition-colors line-clamp-1">
                  {nextArticle.title}
                </div>
              </Link>
            ) : (
              <div />
            )}
          </div>
        </div>

        {/* Right Sticky Sidebar: TOC + Related Tools */}
        <aside className="lg:col-span-4 space-y-6">
          <div className="sticky top-20 space-y-6">
            <TableOfContents items={article.tableOfContents} />

            {/* Related Tools Card */}
            {article.relatedTools.length > 0 && (
              <div className="p-4 rounded-xl border border-border bg-card/60 backdrop-blur-sm space-y-3">
                <div className="text-xs font-semibold text-foreground uppercase tracking-wider">
                  Related CipherVerse Tools
                </div>
                <div className="space-y-2">
                  {article.relatedTools.map((t) => (
                    <Link
                      key={t.path}
                      to={t.path}
                      className="p-2.5 rounded-lg border border-border/60 bg-background/50 hover:bg-secondary/60 hover:border-primary/40 transition-all block group"
                    >
                      <div className="text-xs font-semibold text-foreground group-hover:text-primary transition-colors flex items-center justify-between">
                        <span>{t.name}</span>
                        <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                      </div>
                      <p className="text-[11px] text-muted-foreground line-clamp-2 mt-0.5">
                        {t.description}
                      </p>
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>
    </article>
  );
}
