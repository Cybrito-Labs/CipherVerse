import { z } from 'zod';
import { KeyRound } from 'lucide-react';
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
    description: 'The secret key used for RC2 encryption/decryption.',
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

export default function Rc2Page() {
  return (
    <CipherToolPage
      title="RC2 (Rivest Cipher 2)"
      description="RC2 is a symmetric-key block cipher designed by Ron Rivest in 1987. It was designed to replace DES, but is now considered vulnerable and should only be used for legacy system interoperability."
      icon={KeyRound}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/symmetric/rc2/encrypt"
      decryptEndpoint="/symmetric/rc2/decrypt"
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
