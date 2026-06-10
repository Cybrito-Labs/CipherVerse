import { FileBadge, ShieldCheck, Fingerprint } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const tools = [
  { title: 'X.509 Parser', description: 'Parse and analyze PEM or DER encoded X.509 certificates.', icon: FileBadge, path: '/certificates/x509' },
  { title: 'TLS Analyzer', description: 'Connect to a remote server and analyze its SSL/TLS certificate chain.', icon: ShieldCheck, path: '/certificates/tls' },
  { title: 'Fingerprint Generator', description: 'Generate cryptographic fingerprints (hashes) from raw data or certificates.', icon: Fingerprint, path: '/certificates/fingerprint' },
];

export default function CertificatesPage() {
  return (
    <ToolPageLayout
      title="Certificates & TLS"
      description="Tools for analyzing Public Key Infrastructure (PKI) components, parsing digital certificates, and inspecting TLS connections."
      icon={FileBadge}
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
