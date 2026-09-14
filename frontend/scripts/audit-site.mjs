import fs from 'fs';
import path from 'path';

function getAllHtmlFiles(dir) {
  let results = [];
  const list = fs.readdirSync(dir);
  for (const file of list) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    if (stat && stat.isDirectory()) {
      results = results.concat(getAllHtmlFiles(filePath));
    } else if (file.endsWith('.html')) {
      results.push(filePath);
    }
  }
  return results;
}

console.log('=====================================================');
console.log('    CIPHERVERSE FULL-SITE COMPREHENSIVE AUDIT       ');
console.log('=====================================================\n');

const htmlFiles = getAllHtmlFiles('dist');
console.log(`[1] Crawled HTML Pages: ${htmlFiles.length} static endpoints found.\n`);

let totalJsonLdParsed = 0;
let jsonLdErrors = [];
let schemaTypeCounts = {};
let missingMeta = [];
let totalLinks = 0;
let h1Count = 0;
let missingH1 = [];
let pagesWithDirectories = 0;

for (const f of htmlFiles) {
  const content = fs.readFileSync(f, 'utf8');
  const rel = path.relative('dist', f).replace(/\\/g, '/');

  // 1. Title Tag
  const titleMatch = content.match(/<title>(.*?)<\/title>/);
  if (!titleMatch || !titleMatch[1].trim()) {
    missingMeta.push({ file: rel, issue: 'Missing or empty <title>' });
  }

  // 2. Meta Description
  const descMatch = content.match(/<meta\s+name="description"\s+content="([^"]*)"/);
  if (!descMatch || !descMatch[1].trim()) {
    missingMeta.push({ file: rel, issue: 'Missing or empty meta description' });
  }

  // 3. Canonical Tag
  const canonMatch = content.match(/<link\s+rel="canonical"\s+href="([^"]*)"/);
  if (!canonMatch || !canonMatch[1].trim()) {
    missingMeta.push({ file: rel, issue: 'Missing canonical link' });
  }

  // 4. Open Graph Tags
  const ogTitle = content.match(/<meta\s+property="og:title"\s+content="([^"]*)"/);
  const ogImage = content.match(/<meta\s+property="og:image"\s+content="([^"]*)"/);
  const ogUrl = content.match(/<meta\s+property="og:url"\s+content="([^"]*)"/);
  if (!ogTitle || !ogImage || !ogUrl) {
    missingMeta.push({ file: rel, issue: 'Missing essential OpenGraph tags' });
  }

  // 5. Headings (H1 presence)
  const h1Match = content.match(/<h1[^>]*>([\s\S]*?)<\/h1>/);
  if (h1Match && h1Match[1].trim()) {
    h1Count++;
  } else {
    missingH1.push(rel);
  }

  // 6. JSON-LD scripts (both static template and injected dynamic-seo)
  const scriptRegex = /<script\s+type="application\/ld\+json"[^>]*>([\s\S]*?)<\/script>/gi;
  let match;
  while ((match = scriptRegex.exec(content)) !== null) {
    const rawJson = match[1].trim();
    try {
      const parsed = JSON.parse(rawJson);
      totalJsonLdParsed++;
      if (parsed['@graph']) {
        for (const item of parsed['@graph']) {
          const type = item['@type'] || 'UnknownGraphItem';
          schemaTypeCounts[type] = (schemaTypeCounts[type] || 0) + 1;
        }
      } else if (parsed['@type']) {
        const type = parsed['@type'];
        schemaTypeCounts[type] = (schemaTypeCounts[type] || 0) + 1;
      } else {
        jsonLdErrors.push({ file: rel, error: 'JSON-LD missing @type or @graph' });
      }
    } catch (err) {
      jsonLdErrors.push({ file: rel, error: `Invalid JSON syntax: ${err.message}` });
    }
  }

  // 7. Internal Links & Semantic Shells
  const links = content.match(/href="\/[^"]*"/g);
  if (links) {
    totalLinks += links.length;
  }
  if (
    content.includes('aria-label="Security Suites Directory"') ||
    content.includes('aria-label="Category Tools Directory"') ||
    content.includes('aria-label="Blog Articles Directory"')
  ) {
    pagesWithDirectories++;
  }
}

// 8. Robots.txt, Sitemap.xml, llms.txt, RSS Audit
const publicRobots = fs.existsSync('public/robots.txt');
const distRobots = fs.existsSync('dist/robots.txt');
const publicSitemap = fs.existsSync('public/sitemap.xml');
const distSitemap = fs.existsSync('dist/sitemap.xml');
const publicLlms = fs.existsSync('public/llms.txt');
const distLlms = fs.existsSync('dist/llms.txt');
const publicLlmsFull = fs.existsSync('public/llms-full.txt');
const distLlmsFull = fs.existsSync('dist/llms-full.txt');
const publicRss = fs.existsSync('public/rss.xml');
const distRss = fs.existsSync('dist/rss.xml');
const ogImageExists = fs.existsSync('public/og-image.png');

console.log('--- METADATA & HEAD STATUS ---');
console.log(`Missing Meta or OpenGraph Tags: ${missingMeta.length}`);
if (missingMeta.length > 0) console.log(missingMeta);
console.log(`Pre-rendered <h1> Headings Found: ${h1Count} / ${htmlFiles.length} pages`);
if (missingH1.length > 0) console.log(`Pages missing <h1>:`, missingH1);

console.log('\n--- STRUCTURED DATA (JSON-LD) AUDIT ---');
console.log(`Total Structured Data Blocks Parsed: ${totalJsonLdParsed}`);
console.log(`JSON-LD Syntax / Validation Errors: ${jsonLdErrors.length}`);
if (jsonLdErrors.length > 0) console.log(jsonLdErrors);
console.log('Schema.org Entities Breakdown:');
for (const [type, count] of Object.entries(schemaTypeCounts)) {
  console.log(`  - ${type}: ${count} entities`);
}

console.log('\n--- INTERNAL LINKING & CRAWLABILITY GRAPH ---');
console.log(`Total Pre-rendered Crawlable Hyperlinks: ${totalLinks}`);
console.log(`Pages with Dedicated Directory Navigation Shells: ${pagesWithDirectories}`);

console.log('\n--- ASSETS & PROTOCOLS AUDIT ---');
console.log(`robots.txt:  [Public: ${publicRobots ? 'OK' : 'MISSING'}] [Dist: ${distRobots ? 'OK' : 'MISSING'}]`);
console.log(`sitemap.xml: [Public: ${publicSitemap ? 'OK' : 'MISSING'}] [Dist: ${distSitemap ? 'OK' : 'MISSING'}]`);
console.log(`llms.txt:    [Public: ${publicLlms ? 'OK' : 'MISSING'}] [Dist: ${distLlms ? 'OK' : 'MISSING'}]`);
console.log(`llms-full:   [Public: ${publicLlmsFull ? 'OK' : 'MISSING'}] [Dist: ${distLlmsFull ? 'OK' : 'MISSING'}]`);
console.log(`rss.xml:     [Public: ${publicRss ? 'OK' : 'MISSING'}] [Dist: ${distRss ? 'OK' : 'MISSING'}]`);
console.log(`og-image.png:[Public: ${ogImageExists ? 'OK' : 'MISSING'}]`);

if (publicSitemap) {
  const sitemapXml = fs.readFileSync('public/sitemap.xml', 'utf8');
  const urlCount = (sitemapXml.match(/<loc>/g) || []).length;
  console.log(`Sitemap URLs Indexed: ${urlCount}`);
}

console.log('\n=====================================================');
console.log('               AUDIT COMPLETE                        ');
console.log('=====================================================');
