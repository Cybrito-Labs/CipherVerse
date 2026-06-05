import { z } from 'zod';
import { ShieldAlert } from 'lucide-react';
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
    description: 'The secret key used for DES encryption/decryption.',
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

export default function DesPage() {
  return (
    <CipherToolPage
      title="DES (Data Encryption Standard)"
      description="DES is a legacy symmetric-key block cipher published by NIST. It is now considered insecure for modern applications due to its short 56-bit key length, but remains important for educational and historical purposes."
      icon={ShieldAlert}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/symmetric/des/encrypt"
      decryptEndpoint="/symmetric/des/decrypt"
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
