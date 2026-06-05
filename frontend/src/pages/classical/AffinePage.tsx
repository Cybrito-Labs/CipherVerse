import { z } from 'zod';
import { Calculator } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
  a: z.number().int().min(1),
  b: z.number().int(),
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
    name: 'a',
    label: 'Multiplier (a)',
    type: 'number',
    placeholder: '5',
    defaultValue: 5,
    description: 'Must be coprime to the alphabet size (26).',
  },
  {
    name: 'b',
    label: 'Shift (b)',
    type: 'number',
    placeholder: '8',
    defaultValue: 8,
    description: 'The magnitude of the shift (0-25).',
  },
];

export default function AffinePage() {
  return (
    <CipherToolPage
      title="Affine Cipher"
      description="A type of monoalphabetic substitution cipher, wherein each letter in an alphabet is mapped to its numeric equivalent, encrypted using a simple mathematical function (ax + b) mod 26, and converted back to a letter."
      icon={Calculator}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/classical/affine/encrypt"
      decryptEndpoint="/classical/affine/decrypt"
    />
  );
}
