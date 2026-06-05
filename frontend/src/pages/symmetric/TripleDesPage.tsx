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
    description: 'The secret key used for 3DES encryption/decryption.',
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

export default function TripleDesPage() {
  return (
    <CipherToolPage
      title="3DES (Triple DES)"
      description="Triple DES (3DES) applies the DES cipher algorithm three times to each data block. While more secure than standard DES, it is still considered deprecated for new applications in favor of AES."
      icon={ShieldAlert}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/symmetric/3des/encrypt"
      decryptEndpoint="/symmetric/3des/decrypt"
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
