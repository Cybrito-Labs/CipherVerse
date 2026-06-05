import { z } from 'zod';
import { Layers } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  private_key_hex: z.string().min(64, 'Private key must be at least 64 hex characters').max(64),
  compressed: z.boolean().default(true),
  testnet: z.boolean().default(false),
});

const fields: FieldConfig[] = [
  {
    name: 'private_key_hex',
    label: 'Private Key (Hex)',
    type: 'text',
    placeholder: 'Enter 64-character hex private key...',
    description: 'The raw 256-bit private key in hexadecimal format.',
  },
  {
    name: 'compressed',
    label: 'Compressed Public Key format',
    type: 'checkbox',
    description: 'Generates a WIF that corresponds to a compressed public key (adds 0x01 suffix).',
    isAdvanced: true,
  },
  {
    name: 'testnet',
    label: 'Testnet Network',
    type: 'checkbox',
    description: 'Use the Bitcoin Testnet prefix (0xef) instead of Mainnet (0x80).',
    isAdvanced: true,
  },
];

export default function WifEncoderPage() {
  return (
    <CipherToolPage
      title="WIF Encoder"
      description="Wallet Import Format (WIF) is a standard used to copy and paste private keys easily. This tool encodes a raw hexadecimal private key into base58check WIF."
      icon={Layers}
      fields={fields}
      endpoint="/blockchain/wif/encode"
      schema={schema}
      hasTabs={false}
      transformPayload={(data) => {
        return {
          private_key_hex: data.private_key_hex,
          compressed: data.compressed,
          testnet: data.testnet,
        };
      }}
    />
  );
}
