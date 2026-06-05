import { z } from 'zod';
import { Train } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
  rails: z.number().int().min(2, 'Minimum 2 rails required'),
});

const fields: FieldConfig[] = [
  {
    name: 'text',
    label: 'Text',
    type: 'textarea',
    placeholder: 'Enter text...',
    description: 'The plaintext message to encrypt or ciphertext to decrypt.',
  },
  {
    name: 'rails',
    label: 'Number of Rails',
    type: 'number',
    placeholder: '3',
    defaultValue: 3,
    description: 'The number of rails (depth) used for the zigzag pattern.',
  },
];

export default function RailFencePage() {
  return (
    <CipherToolPage
      title="Rail Fence Cipher"
      description="A transposition cipher that derives its name from the way in which it is encoded. The plaintext is written downwards diagonally on successive 'rails' of an imaginary fence, then moving up when the bottom rail is reached."
      icon={Train}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/classical/rail_fence/encrypt"
      decryptEndpoint="/classical/rail_fence/decrypt"
    />
  );
}
