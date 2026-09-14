import { useState, useMemo } from 'react';
import { blogArticles } from '@/content/blog/articles';
import { BlogCard } from '@/components/blog/BlogCard';
import { Search, BookOpen, ShieldAlert, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

const CATEGORIES = [
  'All',
  'Classical Cryptography',
  'Modern Cryptography',
  'Steganography',
  'Public-Key & Protocols',
] as const;

export default function BlogPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const filteredArticles = useMemo(() => {
    return blogArticles.filter((article) => {
      const matchesCategory =
        selectedCategory === 'All' || article.category === selectedCategory;
      const query = searchQuery.toLowerCase().trim();
      const matchesQuery =
        !query ||
        article.title.toLowerCase().includes(query) ||
        article.description.toLowerCase().includes(query) ||
        article.tags.some((t) => t.toLowerCase().includes(query));
      return matchesCategory && matchesQuery;
    });
  }, [selectedCategory, searchQuery]);

  const featuredArticle = useMemo(() => {
    return blogArticles.find((a) => a.featured) || blogArticles[0];
  }, []);

  const remainingArticles = useMemo(() => {
    if (selectedCategory === 'All' && !searchQuery.trim()) {
      return filteredArticles.filter((a) => a.slug !== featuredArticle?.slug);
    }
    return filteredArticles;
  }, [filteredArticles, selectedCategory, searchQuery, featuredArticle]);

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto space-y-10">

      {/* Hero Header */}
      <div className="text-center max-w-3xl mx-auto space-y-4 pt-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20 shadow-sm">
          <BookOpen className="w-3.5 h-3.5" />
          <span>CipherVerse Knowledge &amp; Research</span>
        </div>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-foreground">
          Cryptography &amp; Cybersecurity <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-cyan-400 to-purple-500">Dispatch</span>
        </h1>
        <p className="text-muted-foreground text-sm sm:text-base max-w-2xl mx-auto leading-relaxed">
          Deep-dive guides, mathematical proofs, algorithm teardowns, and practical tutorials on classical ciphers, modern block ciphers, and information hiding.
        </p>
      </div>

      {/* Search & Category Filter Controls */}
      <div className="space-y-4 max-w-4xl mx-auto">
        <div className="relative max-w-md mx-auto">
          <Search className="w-4 h-4 text-muted-foreground absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search articles by title, topic, or tag..."
            className="w-full pl-10 pr-4 py-2 text-sm rounded-xl border border-border bg-card/80 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary transition-all shadow-sm"
          />
        </div>

        {/* Filter Chips */}
        <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
          {CATEGORIES.map((cat) => {
            const isSelected = selectedCategory === cat;
            return (
              <button
                key={cat}
                type="button"
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                  isSelected
                    ? 'bg-primary text-primary-foreground shadow-sm shadow-primary/20 font-semibold'
                    : 'bg-card text-muted-foreground hover:text-foreground border border-border hover:border-muted-foreground/40'
                }`}
              >
                {cat}
              </button>
            );
          })}
        </div>
      </div>

      {/* Featured Article (when viewing all & no search query) */}
      {selectedCategory === 'All' && !searchQuery.trim() && featuredArticle && (
        <section aria-label="Featured Article" className="space-y-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-primary uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Featured Guide</span>
          </div>
          <BlogCard article={featuredArticle} featured />
        </section>
      )}

      {/* Articles Grid */}
      <section aria-label="Article Catalog" className="space-y-4">
        {remainingArticles.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {remainingArticles.map((article, idx) => (
              <motion.div
                key={article.slug}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay: idx * 0.05 }}
              >
                <BlogCard article={article} />
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="py-16 text-center space-y-3 rounded-2xl border border-dashed border-border bg-card/40 max-w-md mx-auto">
            <ShieldAlert className="w-10 h-10 text-muted-foreground mx-auto" />
            <h3 className="text-base font-semibold text-foreground">No articles found</h3>
            <p className="text-xs text-muted-foreground">
              Try adjusting your search query or selecting a different category.
            </p>
            <button
              onClick={() => {
                setSelectedCategory('All');
                setSearchQuery('');
              }}
              className="mt-2 text-xs font-semibold text-primary hover:underline"
            >
              Reset filters
            </button>
          </div>
        )}
      </section>
    </div>
  );
}
