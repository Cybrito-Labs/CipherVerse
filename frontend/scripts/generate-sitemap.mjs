import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { seoConfigMap, SITE_URL } from '../src/constants/seoConfig.ts';
import { blogArticles } from '../src/content/blog/articles.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicPath = path.resolve(__dirname, '../public/sitemap.xml');
const distPath = path.resolve(__dirname, '../dist/sitemap.xml');

const today = new Date().toISOString().split('T')[0];

const hubRoutes = new Set([
  '/classical',
  '/encoding',
  '/symmetric',
  '/asymmetric',
  '/hashing',
  '/certificates',
  '/blockchain',
  '/steganography',
  '/malware-analysis',
  '/file-forensics',
  '/utilities',
  '/historical',
  '/blog',
]);

const excludedRoutes = new Set(['/settings', '/404', '/api-explorer']);

let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n';

let count = 0;

for (const [route, config] of Object.entries(seoConfigMap)) {
  if (excludedRoutes.has(route)) continue;

  const loc = `${SITE_URL}${route === '/' ? '/' : route}`;
  let priority = '0.8';
  let changefreq = 'weekly';

  if (route === '/') {
    priority = '1.0';
    changefreq = 'daily';
  } else if (hubRoutes.has(route)) {
    priority = '0.9';
    changefreq = 'weekly';
  }

  xml += '  <url>\n';
  xml += `    <loc>${loc}</loc>\n`;
  xml += `    <lastmod>${today}</lastmod>\n`;
  xml += `    <changefreq>${changefreq}</changefreq>\n`;
  xml += `    <priority>${priority}</priority>\n`;
  xml += '  </url>\n';

  count++;
}

// Add individual blog articles
for (const article of blogArticles) {
  const loc = `${SITE_URL}/blog/${article.slug}`;
  xml += '  <url>\n';
  xml += `    <loc>${loc}</loc>\n`;
  xml += `    <lastmod>${article.publishedAt}</lastmod>\n`;
  xml += `    <changefreq>weekly</changefreq>\n`;
  xml += `    <priority>0.8</priority>\n`;
  xml += '  </url>\n';
  count++;
}

xml += '</urlset>\n';

fs.writeFileSync(publicPath, xml, 'utf8');

// Also write to dist if dist exists
if (fs.existsSync(path.dirname(distPath))) {
  fs.writeFileSync(distPath, xml, 'utf8');
}

console.log(`Generated sitemap.xml with ${count} routes (lastmod: ${today})`);
