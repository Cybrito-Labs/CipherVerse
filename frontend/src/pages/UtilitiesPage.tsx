import { Wrench, KeyRound, ShieldCheck, Dice5, FileCode2 } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const tools = [
  { title: 'Password Strength', description: 'Analyze password entropy and receive security suggestions.', icon: ShieldCheck, path: '/utilities/password' },
  { title: 'JWT Sign', description: 'Sign JSON Web Tokens (JWT) with custom payloads and algorithms.', icon: KeyRound, path: '/utilities/jwt' },
  { title: 'Salt Generator', description: 'Generate cryptographically secure random salts.', icon: Dice5, path: '/utilities/salt' },
  { title: 'Fletcher-16 Checksum', description: 'Calculate Fletcher-16 checksums for data integrity verification.', icon: FileCode2, path: '/utilities/fletcher16' },
];

export default function UtilitiesPage() {
  return (
    <ToolPageLayout
      title="Security Utilities"
      description="A collection of essential security tools for developers and analysts."
      icon={Wrench}
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
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
