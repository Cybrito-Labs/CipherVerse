import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { seoConfigMap, SITE_NAME, SITE_URL } from '../src/constants/seoConfig.ts';
import { navigationGroups } from '../src/constants/navigation.ts';
import { getToolKnowledge } from '../src/constants/toolKnowledge.ts';
import { blogArticles } from '../src/content/blog/articles.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.resolve(__dirname, '../public');
const distDir = path.resolve(__dirname, '../dist');

let llmsContent = `# ${SITE_NAME}

> Professional, Zero-Knowledge Cybersecurity & Cryptography Web Suite

CipherVerse is a high-performance, client-side web application offering 40+ interactive cryptographic tools, malware triage utilities, binary forensics analyzers, and historical cipher machine simulators.
All computations run 100% locally in the browser via the Web Crypto API, WebAssembly, and native JavaScript implementations - no private keys, passwords, or encrypted payloads are ever uploaded to a server.

## Canonical Resources
- [Interactive Workspace](${SITE_URL}/): Full browser-based cryptographic suite.
- [Cryptography Blog](${SITE_URL}/blog): In-depth tutorials, mathematical breakdowns, and security teardowns.
- [XML Sitemap](${SITE_URL}/sitemap.xml): Complete machine-readable route directory.
- [API Explorer](${SITE_URL}/api-explorer): REST API documentation and endpoints.

## Cryptography & Cybersecurity Blog Articles
Explore our comprehensive educational research articles:
${blogArticles.map((a) => `- [${a.title}](${SITE_URL}/blog/${a.slug}): ${a.description} (${a.readTime})`).join('\n')}

## Security Suites & Domain Directories
`;

// Group tools by category
const categoryMap = new Map();

for (const group of navigationGroups) {
  for (const item of group.items) {
    if (item.path === '/' || item.path === '/blog') continue;
    categoryMap.set(item.path, {
      name: item.label,
      description: item.description,
      tools: [],
    });
  }
}

for (const [route, config] of Object.entries(seoConfigMap)) {
  if (route === '/' || route === '/settings' || route === '/404' || route === '/api-explorer' || route === '/blog') {
    continue;
  }

  if (route === '/hashing') {
    if (categoryMap.has('/hashing')) {
      categoryMap.get('/hashing').tools.push({
        route,
        title: config.title.split(/[—|]/)[0].trim(),
        description: config.description,
        keywords: config.keywords,
      });
    }
    continue;
  }

  // Check if it's a tool (has a slash after category)
  const segments = route.split('/').filter(Boolean);
  if (segments.length > 1) {
    const parentPath = `/${segments[0]}`;
    if (categoryMap.has(parentPath)) {
      categoryMap.get(parentPath).tools.push({
        route,
        title: config.title.split(/[—|]/)[0].trim(),
        description: config.description,
        keywords: config.keywords,
      });
    }
  }
}

for (const [hubPath, cat] of categoryMap.entries()) {
  llmsContent += `\n### [${cat.name}](${SITE_URL}${hubPath})\n`;
  llmsContent += `${cat.description}\n\n`;

  for (const tool of cat.tools) {
    llmsContent += `- [${tool.title}](${SITE_URL}${tool.route}): ${tool.description}\n`;
  }
}

// Generate llms-full.txt with in-depth technical knowledge, FAQs, blog content, and step-by-step instructions
let llmsFullContent = `${llmsContent}

## Comprehensive Cryptography Blog & Research Articles
${blogArticles
  .map(
    (a) => `
### ${a.title}
URL: ${SITE_URL}/blog/${a.slug}
Author: ${a.author.name} (${a.author.role})
Category: ${a.category} | Reading Time: ${a.readTime}
Description: ${a.description}

Key Sections:
${a.sections.map((s) => `#### ${s.heading}\n${(s.paragraphs || []).join('\n\n')}`).join('\n\n')}
`
  )
  .join('\n---\n')}

## Comprehensive Technical Guide & Algorithmic Details
`;

for (const [hubPath, cat] of categoryMap.entries()) {
  llmsFullContent += `\n---\n\n## Category: ${cat.name}\n`;
  const hubKnowledge = getToolKnowledge(hubPath, cat.name);

  if (hubKnowledge.faqs && hubKnowledge.faqs.length > 0) {
    llmsFullContent += `\n### Frequently Asked Questions\n`;
    for (const faq of hubKnowledge.faqs) {
      llmsFullContent += `- **Q: ${faq.question}**\n  A: ${faq.answer}\n`;
    }
  }

  for (const tool of cat.tools) {
    const knowledge = getToolKnowledge(tool.route, cat.name);
    llmsFullContent += `\n### ${tool.title} (${SITE_URL}${tool.route})\n`;
    llmsFullContent += `${tool.description}\n\n`;

    if (knowledge.howTo && knowledge.howTo.length > 0) {
      llmsFullContent += `**Step-by-Step Instructions:**\n`;
      for (const st of knowledge.howTo) {
        llmsFullContent += `${st.step}. **${st.title}**: ${st.description}\n`;
      }
      llmsFullContent += `\n`;
    }

    if (knowledge.faqs && knowledge.faqs.length > 0) {
      llmsFullContent += `**Technical FAQs:**\n`;
      for (const faq of knowledge.faqs) {
        llmsFullContent += `- **${faq.question}**: ${faq.answer}\n`;
      }
    }
  }
}

function cleanAscii(str) {
  return str
    .replace(/[—–]/g, '-')
    .replace(/[’‘]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/è/g, 'e')
    .replace(/é/g, 'e')
    .replace(/…/g, '...');
}

// Write llms.txt & llms-full.txt to public/ and dist/
const finalLlms = cleanAscii(llmsContent);
const finalLlmsFull = cleanAscii(llmsFullContent);

fs.writeFileSync(path.join(publicDir, 'llms.txt'), finalLlms, 'utf8');
fs.writeFileSync(path.join(publicDir, 'llms-full.txt'), finalLlmsFull, 'utf8');

if (fs.existsSync(distDir)) {
  fs.writeFileSync(path.join(distDir, 'llms.txt'), finalLlms, 'utf8');
  fs.writeFileSync(path.join(distDir, 'llms-full.txt'), finalLlmsFull, 'utf8');
}

console.log('Successfully generated llms.txt and llms-full.txt for Generative Engine Optimization (GEO)!');
