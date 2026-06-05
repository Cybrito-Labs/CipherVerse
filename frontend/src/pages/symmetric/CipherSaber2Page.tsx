import { z } from 'zod';
import { SearchCode } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
  password: z.string().min(1, 'Password is required'),
});

const fields: FieldConfig[] = [
  {
    name: 'text',
    label: 'Text',
    type: 'textarea',
    placeholder: 'Enter text to encrypt or Base64/Hex ciphertext to decrypt...',
    description: 'The plaintext message to encrypt or ciphertext to decrypt.',
  },
  {
    name: 'password',
    label: 'Password',
    type: 'text',
    placeholder: 'Enter secure password...',
    description: 'The secret key used for CipherSaber2 encryption/decryption.',
  },
];

export default function CipherSaber2Page() {
  return (
    <CipherToolPage
      title="CipherSaber2"
      description="CipherSaber is a simple, symmetric stream cipher designed to protect privacy. CipherSaber-2 improves upon the original by adding key stretching via multiple iterations of the RC4 key scheduling algorithm."
      icon={SearchCode}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/symmetric/ciphersaber2/encrypt"
      decryptEndpoint="/symmetric/ciphersaber2/decrypt"
      transformPayload={(data, tab) => {
        if (tab === 'decrypt') {
          return {
            ciphertext: data.text,
            password: data.password,
          };
        }
        return data;
      }}
    />
  );
}
