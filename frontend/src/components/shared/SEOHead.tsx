import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { getSEOConfig, SITE_NAME, SITE_URL, DEFAULT_OG_IMAGE } from '@/constants/seoConfig';

export function SEOHead() {
  const location = useLocation();

  useEffect(() => {
    const seo = getSEOConfig(location.pathname);
    const cleanPath = location.pathname.length > 1 && location.pathname.endsWith('/')
      ? location.pathname.slice(0, -1)
      : location.pathname;
    const currentCanonicalUrl = `${SITE_URL}${cleanPath}`;
    const pageImage = seo.ogImage || DEFAULT_OG_IMAGE;

    // 1. Dynamic Document Title
    document.title = seo.title;

    // Helper function to update or inject meta tags
    const setMetaTag = (selector: string, attributeName: string, attributeValue: string, content: string) => {
      let el = document.querySelector(selector);
      if (!el) {
        el = document.createElement('meta');
        el.setAttribute(attributeName, attributeValue);
        document.head.appendChild(el);
      }
      el.setAttribute('content', content);
    };

    // Helper function to update or inject link tags
    const setLinkTag = (rel: string, href: string) => {
      let el = document.querySelector(`link[rel="${rel}"]`) as HTMLLinkElement | null;
      if (!el) {
        el = document.createElement('link');
        el.setAttribute('rel', rel);
        document.head.appendChild(el);
      }
      el.setAttribute('href', href);
    };

    // 2. Update Primary SEO Meta Tags
    setMetaTag('meta[name="description"]', 'name', 'description', seo.description);
    setMetaTag('meta[name="keywords"]', 'name', 'keywords', seo.keywords.join(', '));
    setMetaTag('meta[name="robots"]', 'name', 'robots', cleanPath === '/404' ? 'noindex, follow' : 'index, follow');

    // 3. Update Canonical URL
    setLinkTag('canonical', currentCanonicalUrl);

    // 4. Update Open Graph Meta Tags
    setMetaTag('meta[property="og:title"]', 'property', 'og:title', seo.title);
    setMetaTag('meta[property="og:description"]', 'property', 'og:description', seo.description);
    setMetaTag('meta[property="og:url"]', 'property', 'og:url', currentCanonicalUrl);
    setMetaTag('meta[property="og:type"]', 'property', 'og:type', cleanPath === '/' ? 'website' : 'article');
    setMetaTag('meta[property="og:site_name"]', 'property', 'og:site_name', SITE_NAME);
    setMetaTag('meta[property="og:image"]', 'property', 'og:image', pageImage);

    // 5. Update Twitter Card Meta Tags
    setMetaTag('meta[name="twitter:card"]', 'name', 'twitter:card', 'summary_large_image');
    setMetaTag('meta[name="twitter:title"]', 'name', 'twitter:title', seo.title);
    setMetaTag('meta[name="twitter:description"]', 'name', 'twitter:description', seo.description);
    setMetaTag('meta[name="twitter:image"]', 'name', 'twitter:image', pageImage);

    // 6. JSON-LD Dynamic Structured Data Generation
    const existingJsonLd = document.querySelectorAll('script[type="application/ld+json"].dynamic-seo');
    existingJsonLd.forEach((s) => s.remove());

    const schemas: object[] = [];

    // BreadcrumbList Schema
    const pathSegments = cleanPath.split('/').filter(Boolean);
    const breadcrumbItems = [
      {
        '@type': 'ListItem',
        position: 1,
        name: 'Home',
        item: SITE_URL,
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
    if (cleanPath !== '/' && cleanPath !== '/404') {
      schemas.push({
        '@context': 'https://schema.org',
        '@type': 'WebApplication',
        name: seo.title.split('—')[0].trim(),
        url: currentCanonicalUrl,
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
    }

    // FAQPage Schema if FAQs exist
    if (seo.faqs && seo.faqs.length > 0) {
      schemas.push({
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        mainEntity: seo.faqs.map((faq) => ({
          '@type': 'Question',
          name: faq.question,
          acceptedAnswer: {
            '@type': 'Answer',
            text: faq.answer,
          },
        })),
      });
    }

    // Inject JSON-LD Script Tags
    schemas.forEach((schemaObj) => {
      const script = document.createElement('script');
      script.type = 'application/ld+json';
      script.className = 'dynamic-seo';
      script.text = JSON.stringify(schemaObj);
      document.head.appendChild(script);
    });

  }, [location.pathname]);

  return null;
}
