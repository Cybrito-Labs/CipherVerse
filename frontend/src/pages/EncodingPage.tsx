import { FileCode, Hash, Link, Binary, Code } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const encodingTools = [
  { title: 'Base64', description: 'Encode and decode Base64 strings. Widely used for encoding binary data as text.', icon: FileCode, path: '/encoding/base64' },
  { title: 'Hex', description: 'Convert text to and from hexadecimal representation.', icon: Hash, path: '/encoding/hex' },
  { title: 'URL', description: 'Percent-encoding for safely transmitting data in URLs.', icon: Link, path: '/encoding/url' },
  { title: 'Binary', description: 'Convert text to binary (base-2) and vice versa.', icon: Binary, path: '/encoding/binary' },
  { title: 'Morse Code', description: 'Translate text to standard Morse code dots and dashes.', icon: Code, path: '/encoding/morse' },
];

export default function EncodingPage() {
  return (
    <ToolPageLayout
      title="Encoding & Decoding"
      description="Tools for converting data into different formats and representations. These are not encryption algorithms, but standardized data transformations."
      icon={FileCode}
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {encodingTools.map((tool) => (
          <ToolCard
            key={tool.path}
            title={tool.title}
            description={tool.description}
            icon={tool.icon}
            path={tool.path}
          />
        ))}
      </div>
    </ToolPageLayout>
  );
}
