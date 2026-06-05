import { z } from 'zod';
import { Lock } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
  password: z.string().min(1, 'Password is required'),
  mode: z.enum(['ECB', 'CBC', 'CFB', 'OFB']),
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
    description: 'The secret key used for Blowfish encryption/decryption.',
  },
  {
    name: 'mode',
    label: 'Block Cipher Mode',
    type: 'select',
    placeholder: 'Select mode...',
    defaultValue: 'CBC',
    description: 'The mode of operation for the block cipher.',
    isAdvanced: true,
    options: [
      { label: 'CBC (Cipher Block Chaining)', value: 'CBC' },
      { label: 'ECB (Electronic Codebook)', value: 'ECB' },
      { label: 'CFB (Cipher Feedback)', value: 'CFB' },
      { label: 'OFB (Output Feedback)', value: 'OFB' },
    ],
  },
];

export default function BlowfishPage() {
  return (
    <CipherToolPage
      title="Blowfish"
      description="Blowfish is a symmetric-key block cipher designed in 1993 by Bruce Schneier. Known for its speed and effectiveness, it was created as a general-purpose algorithm and intended as an alternative to the aging DES."
      icon={Lock}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/symmetric/blowfish/encrypt"
      decryptEndpoint="/symmetric/blowfish/decrypt"
      transformPayload={(data, tab) => {
        if (tab === 'decrypt') {
          return {
            ciphertext: data.text,
            password: data.password,
            mode: data.mode,
          };
        }
        return data;
      }}
    />
  );
}
