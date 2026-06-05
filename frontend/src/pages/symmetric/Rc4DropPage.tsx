import { z } from 'zod';
import { KeyRound } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text/Data is required'),
  key: z.string().min(1, 'Key is required'),
  drop_n: z.number().int().min(0),
});

const fields: FieldConfig[] = [
  {
    name: 'text',
    label: 'Text',
    type: 'textarea',
    placeholder: 'Enter plaintext to encrypt or Hex ciphertext to decrypt...',
    description: 'The plaintext message to encrypt or ciphertext to decrypt.',
  },
  {
    name: 'key',
    label: 'Key',
    type: 'text',
    placeholder: 'Enter secure key...',
    description: 'The secret key used for RC4 Drop encryption/decryption.',
  },
  {
    name: 'drop_n',
    label: 'Drop Bytes',
    type: 'number',
    placeholder: '768',
    defaultValue: 768,
    description: 'Number of initial keystream bytes to discard (e.g., 768 or 3072).',
    isAdvanced: true,
  },
];

export default function Rc4DropPage() {
  return (
    <CipherToolPage
      title="RC4 Drop"
      description="RC4 Drop modifies the standard RC4 algorithm by discarding the initial portion of the keystream (often the first 768 or 3072 bytes). This helps mitigate attacks that exploit the weak correlation between the key and the initial keystream bytes."
      icon={KeyRound}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/symmetric/rc4/drop/encrypt"
      decryptEndpoint="/symmetric/rc4/drop/decrypt"
    />
  );
}
