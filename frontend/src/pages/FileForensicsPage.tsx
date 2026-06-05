import { Search, Fingerprint, Activity, Binary, FileSearch } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const tools = [
  { title: 'File Hash', description: 'Calculate a single cryptographic hash of a local file.', icon: Fingerprint, path: '/file-forensics/hash' },
  { title: 'Multi Hash', description: 'Calculate multiple hashes (MD5, SHA1, SHA256) simultaneously for comprehensive file identification.', icon: Search, path: '/file-forensics/multi-hash' },
  { title: 'Shannon Entropy', description: 'Calculate file entropy to detect encrypted or compressed (packed) data.', icon: Activity, path: '/file-forensics/entropy' },
  { title: 'Randomness Test', description: 'Perform statistical randomness tests to evaluate file contents for cryptographic strength.', icon: Binary, path: '/file-forensics/randomness' },
];

export default function FileForensicsPage() {
  return (
    <ToolPageLayout
      title="File Forensics"
      description="Analyze files on disk to calculate cryptographic hashes, measure entropy, and perform statistical randomness tests."
      icon={FileSearch}
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {tools.map((tool) => (
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
