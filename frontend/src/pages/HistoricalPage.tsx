import { RotateCw, Search, Calculator, Archive } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const tools = [
  { title: 'Enigma Machine', description: 'Simulate the legendary WWII German Enigma machine with custom rotor settings.', icon: RotateCw, path: '/historical/enigma' },
  { title: 'Bombe Simulator', description: 'Simulate Alan Turing\'s Bombe to cryptanalyze Enigma ciphertext using known plaintext cribs.', icon: Search, path: '/historical/bombe' },
  { title: 'Typex Machine', description: 'Simulate the British Typex cipher machine, a highly modified and more secure Enigma variant.', icon: Calculator, path: '/historical/typex' },
];

export default function HistoricalPage() {
  return (
    <ToolPageLayout
      title="Historical Machines"
      description="Interactive simulations of famous electro-mechanical cryptographic machines from World War II."
      icon={Archive}
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
