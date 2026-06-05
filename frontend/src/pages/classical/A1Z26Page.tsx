import { z } from 'zod';
import { Hash } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
});

const fields: FieldConfig[] = [
  {
    name: 'text',
    label: 'Text',
    type: 'textarea',
    placeholder: 'Enter text...',
    description: 'The plaintext message to encode or numbers to decode (e.g. 8-5-12-12-15).',
  },
];

export default function A1Z26Page() {
  return (
    <CipherToolPage
      title="A1Z26 Cipher"
      description="A very simple substitution cipher that replaces each letter with its corresponding number in the alphabet (A=1, B=2, ..., Z=26)."
      icon={Hash}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/classical/a1z26/encrypt"
      decryptEndpoint="/classical/a1z26/decrypt"
    />
  );
}
