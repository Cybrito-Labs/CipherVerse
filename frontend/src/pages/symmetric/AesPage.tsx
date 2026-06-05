import { z } from 'zod';
import { ShieldCheck } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
  password: z.string().min(1, 'Password is required'),
  mode: z.enum(['ECB', 'CBC', 'CFB', 'OFB', 'CTR']),
  key_size: z.number().int().min(128).max(256),
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
    description: 'The secret key used for AES encryption/decryption.',
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
      { label: 'CTR (Counter)', value: 'CTR' },
    ],
  },
  {
    name: 'key_size',
    label: 'Key Size (bits)',
    type: 'select',
    placeholder: 'Select key size...',
    defaultValue: 256,
    description: 'AES key size. 256-bit is recommended for maximum security.',
    isAdvanced: true,
    options: [
      { label: '128-bit', value: '128' },
      { label: '192-bit', value: '192' },
      { label: '256-bit', value: '256' },
    ],
  },
];

export default function AesPage() {
  return (
    <CipherToolPage
      title="AES (Advanced Encryption Standard)"
      description="The Advanced Encryption Standard (AES) is a symmetric block cipher established by the U.S. NIST. It is widely used across the globe to protect sensitive data."
      icon={ShieldCheck}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/symmetric/aes/encrypt"
      decryptEndpoint="/symmetric/aes/decrypt"
      transformPayload={(data, tab) => {
        // Ensure key_size is sent as a number since select fields return strings
        const payload = { ...data, key_size: Number(data.key_size) };
        if (tab === 'decrypt') {
          return {
            ciphertext: payload.text,
            password: payload.password,
            mode: payload.mode,
            key_size: payload.key_size,
          };
        }
        return payload;
      }}
    />
  );
}
