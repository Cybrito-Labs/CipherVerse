import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SITE_NAME, SITE_URL } from '../src/constants/seoConfig.ts';
import { blogArticles } from '../src/content/blog/articles.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicPath = path.resolve(__dirname, '../public/rss.xml');
const distPath = path.resolve(__dirname, '../dist/rss.xml');

function escapeXml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

const buildDate = new Date().toUTCString();

let rss = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${escapeXml(SITE_NAME)} — Cryptography &amp; Cybersecurity Blog</title>
    <link>${SITE_URL}/blog</link>
    <description>In-depth cryptography tutorials, mathematical breakdowns, steganography techniques, and cryptanalysis guides.</description>
    <language>en-us</language>
    <lastBuildDate>${buildDate}</lastBuildDate>
    <atom:link href="${SITE_URL}/rss.xml" rel="self" type="application/rss+xml" />
`;

for (const article of blogArticles) {
  const articleUrl = `${SITE_URL}/blog/${article.slug}`;
  const pubDate = new Date(article.publishedAt).toUTCString();

  rss += `    <item>
      <title>${escapeXml(article.title)}</title>
      <link>${articleUrl}</link>
      <guid isPermaLink="true">${articleUrl}</guid>
      <description>${escapeXml(article.description)}</description>
      <category>${escapeXml(article.category)}</category>
      <author>${escapeXml(article.author.name)}</author>
      <pubDate>${pubDate}</pubDate>
    </item>
`;
}

rss += `  </channel>
</rss>
`;

fs.writeFileSync(publicPath, rss, 'utf8');

const distDir = path.resolve(__dirname, '../dist');
if (fs.existsSync(distDir)) {
  fs.writeFileSync(distPath, rss, 'utf8');
}

console.log(`Successfully generated rss.xml with ${blogArticles.length} blog articles!`);
