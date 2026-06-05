import { Bitcoin, Layers, Network, Wallet } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const tools = [
  { title: 'Bitcoin Validation', description: 'Validate Bitcoin addresses (P2PKH, P2SH, Bech32).', icon: Bitcoin, path: '/blockchain/bitcoin' },
  { title: 'Ethereum Validation', description: 'Validate Ethereum addresses and checksums (EIP-55).', icon: Wallet, path: '/blockchain/ethereum' },
  { title: 'Merkle Tree', description: 'Construct and visualize cryptographic Merkle trees.', icon: Network, path: '/blockchain/merkle' },
  { title: 'WIF Encoder', description: 'Encode private keys into Wallet Import Format (WIF).', icon: Layers, path: '/blockchain/wif' },
];

export default function BlockchainPage() {
  return (
    <ToolPageLayout
      title="Blockchain Tools"
      description="Cryptographic tools for interacting with and analyzing blockchain structures, addresses, and key formats."
      icon={Bitcoin}
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
