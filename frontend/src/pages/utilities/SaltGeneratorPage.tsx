import { z } from 'zod';
import { Dice5 } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  length: z.coerce.number().min(8, 'Length must be at least 8').max(128, 'Length cannot exceed 128'),
});

const fields: FieldConfig[] = [
  {
    name: 'length',
    label: 'Salt Length (bytes)',
    type: 'number',
    placeholder: 'e.g., 16',
    defaultValue: 16,
    description: 'The number of random bytes to generate. A standard secure salt length is 16 bytes (which produces a 32-character hex string).',
  },
];

export default function SaltGeneratorPage() {
  return (
    <CipherToolPage
      title="Cryptographic Salt Generator"
      description="Generate cryptographically secure random bytes for use as password salts, IVs (Initialization Vectors), or nonces."
      icon={Dice5}
      fields={fields}
      endpoint="/utilities/salt/generate"
      schema={schema}
      hasTabs={false}
      transformPayload={(data) => ({ length: data.length })}
    />
  );
}
