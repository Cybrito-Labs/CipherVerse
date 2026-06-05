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
    description: 'The secret key used for SM4 encryption/decryption.',
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

export default function Sm4Page() {
  return (
    <CipherToolPage
      title="SM4"
      description="SM4 is a block cipher used in the Chinese National Standard for Wireless LAN WAPI. It was established as a national standard in 2006 and is widely used in Chinese government and commercial applications."
      icon={Lock}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/symmetric/sm4/encrypt"
      decryptEndpoint="/symmetric/sm4/decrypt"
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
