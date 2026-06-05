import { z } from 'zod';
import { KeyRound } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text/Data is required'),
  key: z.string().min(1, 'Key is required'),
});

const fields: FieldConfig[] = [
  {
    name: 'text',
    label: 'Text / Hex Data',
    type: 'textarea',
    placeholder: 'Enter plaintext to encrypt or Hex ciphertext to decrypt...',
    description: 'The plaintext message to encrypt or hex-encoded ciphertext to decrypt.',
  },
  {
    name: 'key',
    label: 'Key',
    type: 'text',
    placeholder: 'Enter secure key...',
    description: 'The secret key used for RC4 stream encryption/decryption.',
  },
];

export default function Rc4Page() {
  return (
    <CipherToolPage
      title="RC4 (Rivest Cipher 4)"
      description="RC4 is a stream cipher designed by Ron Rivest. While remarkable for its simplicity and speed in software, multiple vulnerabilities have been discovered in RC4, rendering it insecure for modern use."
      icon={KeyRound}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/symmetric/rc4/encrypt"
      decryptEndpoint="/symmetric/rc4/decrypt"
      transformPayload={(data, tab) => {
        if (tab === 'decrypt') {
          return {
            hex_data: data.text,
            key: data.key,
          };
        }
        return data;
      }}
    />
  );
}
