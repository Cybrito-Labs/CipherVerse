import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  seoConfigMap,
  SITE_NAME,
  SITE_URL,
  DEFAULT_OG_IMAGE,
  DEFAULT_KEYWORDS,
  getSEOConfig,
} from '../src/constants/seoConfig.ts';
import { getToolKnowledge } from '../src/constants/toolKnowledge.ts';
import { navigationGroups } from '../src/constants/navigation.ts';
import { blogArticles } from '../src/content/blog/articles.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.resolve(__dirname, '../dist');
const templatePath = path.join(distDir, 'index.html');

if (!fs.existsSync(templatePath)) {
  console.error('Error: dist/index.html not found. Run vite build first.');
  process.exit(1);
}

const template = fs.readFileSync(templatePath, 'utf8');

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

let generatedCount = 0;

for (const [route, seo] of Object.entries(seoConfigMap)) {
  if (route === '/settings' || route === '/404') {
    continue; // Don't pre-render private or 404 routes
  }

  const isHome = route === '/';
  const cleanPath = isHome ? '' : route;
  const canonicalUrl = isHome ? `${SITE_URL}/` : `${SITE_URL}${cleanPath}`;
  const pageImage = seo.ogImage || DEFAULT_OG_IMAGE;
  const pageTitle = escapeHtml(seo.title);
  const pageDescription = escapeHtml(seo.description);
  const pageKeywords = escapeHtml((seo.keywords || DEFAULT_KEYWORDS).join(', '));

  // Retrieve unique tool/category knowledge (step-by-step how-to & FAQs)
  const knowledge = getToolKnowledge(route, seo.category);
  const howTo = knowledge.howTo;
  const faqs = (seo.faqs && seo.faqs.length > 0) ? seo.faqs : knowledge.faqs;
  const rawToolName = seo.title.split(/[—|]/)[0].trim();
  const toolName = escapeHtml(rawToolName);

  // 1. Build JSON-LD structured data schemas
  const schemas = [];

  // BreadcrumbList Schema
  const pathSegments = cleanPath.split('/').filter(Boolean);
  const breadcrumbItems = [
    {
      '@type': 'ListItem',
      position: 1,
      name: 'Home',
      item: `${SITE_URL}/`,
    },
  ];

  let accumulatedPath = '';
  pathSegments.forEach((segment, idx) => {
    accumulatedPath += `/${segment}`;
    const segmentSeo = getSEOConfig(accumulatedPath);
    const segmentName = segmentSeo.category || segment.charAt(0).toUpperCase() + segment.slice(1).replace(/-/g, ' ');
    breadcrumbItems.push({
      '@type': 'ListItem',
      position: idx + 2,
      name: segmentName,
      item: `${SITE_URL}${accumulatedPath}`,
    });
  });

  schemas.push({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: breadcrumbItems,
  });

  // WebApplication Schema for Tool Pages
  if (!isHome) {
    schemas.push({
      '@context': 'https://schema.org',
      '@type': 'WebApplication',
      name: rawToolName,
      url: canonicalUrl,
      description: seo.description,
      applicationCategory: 'SecurityApplication',
      operatingSystem: 'All',
      browserRequirements: 'Requires JavaScript. Requires HTML5.',
      offers: {
        '@type': 'Offer',
        price: '0',
        priceCurrency: 'USD',
      },
    });

    // HowTo Schema for Step-by-Step guides
    if (howTo && howTo.length > 0) {
      schemas.push({
        '@context': 'https://schema.org',
        '@type': 'HowTo',
        name: knowledge.guideTitle || `How to Use ${rawToolName}`,
        description: knowledge.guideSubtitle || `Step-by-step cryptographic instructions for using ${rawToolName} on CipherVerse.`,
        step: howTo.map((st) => ({
          '@type': 'HowToStep',
          position: st.step,
          name: st.title,
          text: st.description,
        })),
      });
    }
  }

  // FAQPage Schema if FAQs exist
  if (faqs && faqs.length > 0) {
    schemas.push({
      '@context': 'https://schema.org',
      '@type': 'FAQPage',
      mainEntity: faqs.map((faq) => ({
        '@type': 'Question',
        name: faq.question,
        acceptedAnswer: {
          '@type': 'Answer',
          text: faq.answer,
        },
      })),
    });
  }

  const jsonLdHtml = schemas
    .map((s) => `<script type="application/ld+json" class="dynamic-seo">${JSON.stringify(s)}</script>`)
    .join('\n    ');

  // 2. Semantic Crawlable Content Shell inside #root with unique guide & FAQs
  let semanticShell = `
    <header style="max-width:1200px;margin:2rem auto 1rem;padding:0 1.5rem;font-family:system-ui,sans-serif;">
      <h1 style="font-size:2rem;font-weight:700;color:#f8fafc;margin-bottom:0.5rem;">${toolName}</h1>
      <p style="font-size:1.05rem;color:#94a3b8;line-height:1.6;max-width:48rem;">${pageDescription}</p>
    </header>`;

  // For Homepage: Render all security suite categories for instant crawler discovery
  if (isHome) {
    const suites = navigationGroups.flatMap((g) => g.items.filter((item) => item.path !== '/'));
    semanticShell += `
    <nav aria-label="Security Suites Directory" style="max-width:1200px;margin:2rem auto;padding:0 1.5rem;font-family:system-ui,sans-serif;">
      <h2 style="font-size:1.5rem;font-weight:600;color:#f8fafc;margin-bottom:1rem;">Cybersecurity &amp; Cryptography Suites</h2>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem;">
        ${suites
          .map(
            (suite) => `
          <a href="${suite.path}" style="display:block;padding:1.25rem;border-radius:0.75rem;background:#0f172a;border:1px solid #1e293b;text-decoration:none;color:inherit;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
              <h3 style="font-size:1.1rem;font-weight:600;color:#38bdf8;margin:0;">${escapeHtml(suite.label)}</h3>
              ${suite.toolCount ? `<span style="font-size:0.75rem;padding:0.15rem 0.5rem;border-radius:9999px;background:#1e293b;color:#94a3b8;">${suite.toolCount} tools</span>` : ''}
            </div>
            <p style="font-size:0.875rem;color:#94a3b8;line-height:1.5;margin:0;">${escapeHtml(suite.description)}</p>
          </a>`
          )
          .join('')}
      </div>
    </nav>`;
  }

  // For Blog Index Page: Render all articles for instant crawler indexing
  if (route === '/blog') {
    semanticShell += `
    <nav aria-label="Blog Articles Directory" style="max-width:1200px;margin:2rem auto;padding:0 1.5rem;font-family:system-ui,sans-serif;">
      <h2 style="font-size:1.5rem;font-weight:600;color:#f8fafc;margin-bottom:1rem;">All Cryptography &amp; Cybersecurity Articles</h2>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1.25rem;">
        ${blogArticles
          .map(
            (article) => `
          <a href="/blog/${article.slug}" style="display:block;padding:1.25rem;border-radius:0.75rem;background:#0f172a;border:1px solid #1e293b;text-decoration:none;color:inherit;">
            <span style="font-size:0.75rem;color:#38bdf8;font-weight:600;">${escapeHtml(article.category)}</span>
            <h3 style="font-size:1.15rem;font-weight:600;color:#f8fafc;margin:0.35rem 0;">${escapeHtml(article.title)}</h3>
            <p style="font-size:0.875rem;color:#94a3b8;line-height:1.5;margin:0;">${escapeHtml(article.description)}</p>
            <div style="font-size:0.75rem;color:#64748b;margin-top:0.75rem;">${article.publishedAt} &bull; ${article.readTime}</div>
          </a>`
          )
          .join('')}
      </div>
    </nav>`;
  }

  // For Category Hub Pages: Render direct links to all child tools
  const categoryTools = Object.entries(seoConfigMap).filter(
    ([r]) => r.startsWith(`${route}/`) && r !== route && r !== '/settings' && r !== '/404'
  );

  if (categoryTools.length > 0) {
    semanticShell += `
    <nav aria-label="Category Tools Directory" style="max-width:1200px;margin:2rem auto;padding:0 1.5rem;font-family:system-ui,sans-serif;">
      <h2 style="font-size:1.5rem;font-weight:600;color:#f8fafc;margin-bottom:1rem;">Included Tools &amp; Solvers</h2>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem;">
        ${categoryTools
          .map(
            ([r, cfg]) => `
          <a href="${r}" style="display:block;padding:1rem 1.25rem;border-radius:0.75rem;background:#0f172a;border:1px solid #1e293b;text-decoration:none;color:inherit;">
            <h3 style="font-size:1.05rem;font-weight:600;color:#38bdf8;margin:0 0 0.35rem;">${escapeHtml(cfg.title.split(/[—|]/)[0].trim())}</h3>
            <p style="font-size:0.85rem;color:#94a3b8;line-height:1.5;margin:0;">${escapeHtml(cfg.description)}</p>
          </a>`
          )
          .join('')}
      </div>
    </nav>`;
  }

  // Dedicated Companion Educational Guide Banner for static crawling
  const companionArticle = !isHome && route !== '/blog'
    ? blogArticles.find(
        (article) =>
          article.relatedTools.some((t) => t.path === route) &&
          (article.slug.includes(route.split('/').pop() || '') ||
            article.relatedTools[0]?.path === route)
      )
    : null;

  if (companionArticle) {
    semanticShell += `
    <section aria-label="Companion Educational Guide" style="max-width:1200px;margin:1.5rem auto;padding:1.5rem;border-radius:1rem;background:linear-gradient(to right, rgba(56,189,248,0.1), rgba(15,23,42,0.6));border:1px solid rgba(56,189,248,0.3);font-family:system-ui,sans-serif;">
      <span style="font-size:0.75rem;font-weight:600;color:#38bdf8;text-transform:uppercase;letter-spacing:0.05em;">Deep-Dive Educational Guide</span>
      <h2 style="font-size:1.4rem;font-weight:700;color:#f8fafc;margin:0.35rem 0;">${escapeHtml(companionArticle.title)}</h2>
      <p style="font-size:0.95rem;color:#94a3b8;line-height:1.6;margin-bottom:1rem;">${escapeHtml(companionArticle.description)}</p>
      <a href="/blog/${companionArticle.slug}" style="display:inline-flex;align-items:center;gap:0.5rem;font-size:0.9rem;font-weight:600;color:#38bdf8;text-decoration:none;">Read Companion Guide &rarr;</a>
    </section>`;
  }

  if (!isHome && howTo && howTo.length > 0) {
    semanticShell += `
    <section style="max-width:1200px;margin:2rem auto 1rem;padding:0 1.5rem;font-family:system-ui,sans-serif;">
      <h2 style="font-size:1.5rem;font-weight:600;color:#f8fafc;margin-bottom:1rem;">${escapeHtml(knowledge.guideTitle || `How to Use ${toolName}`)}</h2>
      <ol style="padding-left:1.25rem;color:#cbd5e1;line-height:1.7;">
        ${howTo
          .map(
            (st) => `
          <li style="margin-bottom:0.75rem;">
            <strong style="color:#f1f5f9;">${escapeHtml(st.title)}:</strong> ${escapeHtml(st.description)}
          </li>`
          )
          .join('')}
      </ol>
    </section>`;
  }

  if (faqs && faqs.length > 0) {
    semanticShell += `
    <section style="max-width:1200px;margin:2rem auto;padding:0 1.5rem;font-family:system-ui,sans-serif;">
      <h2 style="font-size:1.5rem;font-weight:600;color:#f8fafc;margin-bottom:1rem;">Frequently Asked Questions</h2>
      ${faqs
        .map(
          (faq) => `
        <article style="margin-bottom:1.25rem;">
          <h3 style="font-size:1.15rem;font-weight:600;color:#38bdf8;margin-bottom:0.25rem;">${escapeHtml(faq.question)}</h3>
          <p style="color:#cbd5e1;line-height:1.6;">${escapeHtml(faq.answer)}</p>
        </article>`
        )
        .join('')}
    </section>`;
  }

  // 3. Replace Metadata in Base Template
  let pageHtml = template;

  // Title
  pageHtml = pageHtml.replace(/<title>.*?<\/title>/, `<title>${pageTitle}</title>`);

  // Meta Description
  pageHtml = pageHtml.replace(
    /<meta name="description" content=".*?" \/>/,
    `<meta name="description" content="${pageDescription}" />`
  );

  // Meta Keywords
  pageHtml = pageHtml.replace(
    /<meta name="keywords" content=".*?" \/>/,
    `<meta name="keywords" content="${pageKeywords}" />`
  );

  // Canonical
  pageHtml = pageHtml.replace(
    /<link rel="canonical" href=".*?" \/>/,
    `<link rel="canonical" href="${canonicalUrl}" />`
  );

  // Open Graph
  pageHtml = pageHtml.replace(
    /<meta property="og:title" content=".*?" \/>/,
    `<meta property="og:title" content="${pageTitle}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta property="og:description" content=".*?" \/>/,
    `<meta property="og:description" content="${pageDescription}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta property="og:url" content=".*?" \/>/,
    `<meta property="og:url" content="${canonicalUrl}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta property="og:type" content=".*?" \/>/,
    `<meta property="og:type" content="${isHome ? 'website' : 'article'}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta property="og:image" content=".*?" \/>/,
    `<meta property="og:image" content="${pageImage}" />`
  );

  // Twitter Cards
  pageHtml = pageHtml.replace(
    /<meta name="twitter:title" content=".*?" \/>/,
    `<meta name="twitter:title" content="${pageTitle}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta name="twitter:description" content=".*?" \/>/,
    `<meta name="twitter:description" content="${pageDescription}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta name="twitter:url" content=".*?" \/>/,
    `<meta name="twitter:url" content="${canonicalUrl}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta name="twitter:image" content=".*?" \/>/,
    `<meta name="twitter:image" content="${pageImage}" />`
  );

  // Inject JSON-LD right before </head>
  pageHtml = pageHtml.replace('</head>', `    ${jsonLdHtml}\n  </head>`);

  // Inject semantic crawlable shell inside <div id="root"></div>
  pageHtml = pageHtml.replace('<div id="root"></div>', `<div id="root">${semanticShell}</div>`);

  // 4. Write HTML file to appropriate directory
  if (isHome) {
    fs.writeFileSync(templatePath, pageHtml, 'utf8');
  } else {
    const targetDir = path.join(distDir, route.replace(/^\//, ''));
    fs.mkdirSync(targetDir, { recursive: true });
    fs.writeFileSync(path.join(targetDir, 'index.html'), pageHtml, 'utf8');
  }

  generatedCount++;
}

// 5. Pre-render individual Blog Article pages with Schema.org BlogPosting
for (const article of blogArticles) {
  const route = `/blog/${article.slug}`;
  const canonicalUrl = `${SITE_URL}${route}`;
  const pageTitle = escapeHtml(`${article.title} | CipherVerse Blog`);
  const pageDescription = escapeHtml(article.description);
  const pageKeywords = escapeHtml(article.tags.join(', '));
  const pageImage = DEFAULT_OG_IMAGE;

  // JSON-LD Schemas: Breadcrumbs and BlogPosting
  const schemas = [
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        {
          '@type': 'ListItem',
          position: 1,
          name: 'Home',
          item: `${SITE_URL}/`,
        },
        {
          '@type': 'ListItem',
          position: 2,
          name: 'Blog',
          item: `${SITE_URL}/blog`,
        },
        {
          '@type': 'ListItem',
          position: 3,
          name: article.title,
          item: canonicalUrl,
        },
      ],
    },
    {
      '@context': 'https://schema.org',
      '@type': 'BlogPosting',
      headline: article.title,
      description: article.description,
      url: canonicalUrl,
      datePublished: article.publishedAt,
      dateModified: article.publishedAt,
      author: {
        '@type': 'Person',
        name: article.author.name,
      },
      publisher: {
        '@type': 'Organization',
        name: SITE_NAME,
        url: SITE_URL,
      },
      mainEntityOfPage: {
        '@type': 'WebPage',
        '@id': canonicalUrl,
      },
      keywords: article.tags.join(', '),
    },
  ];

  const jsonLdHtml = schemas
    .map((s) => `<script type="application/ld+json" class="dynamic-seo">${JSON.stringify(s)}</script>`)
    .join('\n    ');

  let semanticShell = `
    <article style="max-width:900px;margin:2rem auto;padding:0 1.5rem;font-family:system-ui,sans-serif;">
      <header style="margin-bottom:2rem;">
        <span style="font-size:0.85rem;color:#38bdf8;font-weight:600;">${escapeHtml(article.category)}</span>
        <h1 style="font-size:2.25rem;font-weight:800;color:#f8fafc;margin:0.5rem 0;">${escapeHtml(article.title)}</h1>
        <p style="font-size:1.1rem;color:#94a3b8;line-height:1.6;">${escapeHtml(article.description)}</p>
        <div style="font-size:0.85rem;color:#64748b;margin-top:0.75rem;">
          By ${escapeHtml(article.author.name)} &bull; ${article.publishedAt} &bull; ${article.readTime}
        </div>
      </header>
      ${article.sections
        .map(
          (s) => `
        <section style="margin-bottom:2rem;">
          <h2 style="font-size:1.5rem;font-weight:700;color:#f8fafc;margin-bottom:0.75rem;">${escapeHtml(s.heading)}</h2>
          ${(s.paragraphs || []).map((p) => `<p style="font-size:1rem;color:#cbd5e1;line-height:1.7;margin-bottom:1rem;">${escapeHtml(p)}</p>`).join('')}
          ${(s.postCodeParagraphs || []).map((p) => `<p style="font-size:1rem;color:#cbd5e1;line-height:1.7;margin-bottom:1rem;">${escapeHtml(p)}</p>`).join('')}
          ${
            s.toolCta
              ? `
          <div style="margin:1.5rem 0;padding:1.25rem;background:#0f172a;border:1px solid #1e293b;border-radius:0.75rem;">
            <a href="${s.toolCta.path}" style="color:#38bdf8;font-weight:600;text-decoration:none;font-size:1.05rem;">Try ${escapeHtml(s.toolCta.name)} &rarr;</a>
            <p style="color:#94a3b8;font-size:0.875rem;margin:0.35rem 0 0;">${escapeHtml(s.toolCta.description)}</p>
          </div>`
              : ''
          }
        </section>`
        )
        .join('')}
    </article>`;

  let pageHtml = template;
  pageHtml = pageHtml.replace(/<title>.*?<\/title>/, `<title>${pageTitle}</title>`);
  pageHtml = pageHtml.replace(
    /<meta name="description" content=".*?" \/>/,
    `<meta name="description" content="${pageDescription}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta name="keywords" content=".*?" \/>/,
    `<meta name="keywords" content="${pageKeywords}" />`
  );
  pageHtml = pageHtml.replace(
    /<link rel="canonical" href=".*?" \/>/,
    `<link rel="canonical" href="${canonicalUrl}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta property="og:title" content=".*?" \/>/,
    `<meta property="og:title" content="${pageTitle}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta property="og:description" content=".*?" \/>/,
    `<meta property="og:description" content="${pageDescription}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta property="og:url" content=".*?" \/>/,
    `<meta property="og:url" content="${canonicalUrl}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta property="og:type" content=".*?" \/>/,
    `<meta property="og:type" content="article" />`
  );
  pageHtml = pageHtml.replace(
    /<meta property="og:image" content=".*?" \/>/,
    `<meta property="og:image" content="${pageImage}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta name="twitter:title" content=".*?" \/>/,
    `<meta name="twitter:title" content="${pageTitle}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta name="twitter:description" content=".*?" \/>/,
    `<meta name="twitter:description" content="${pageDescription}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta name="twitter:url" content=".*?" \/>/,
    `<meta name="twitter:url" content="${canonicalUrl}" />`
  );
  pageHtml = pageHtml.replace(
    /<meta name="twitter:image" content=".*?" \/>/,
    `<meta name="twitter:image" content="${pageImage}" />`
  );
  pageHtml = pageHtml.replace('</head>', `    ${jsonLdHtml}\n  </head>`);
  pageHtml = pageHtml.replace('<div id="root"></div>', `<div id="root">${semanticShell}</div>`);

  const targetDir = path.join(distDir, 'blog', article.slug);
  fs.mkdirSync(targetDir, { recursive: true });
  fs.writeFileSync(path.join(targetDir, 'index.html'), pageHtml, 'utf8');
  generatedCount++;
}

console.log(`Successfully pre-rendered unique SEO pages for ${generatedCount} routes (including blog articles)!`);
