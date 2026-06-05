import { z } from 'zod';
import { Grid3X3 } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
  keyword: z.string().min(1, 'Keyword is required'),
});

const fields: FieldConfig[] = [
  {
    name: 'text',
    label: 'Text',
    type: 'textarea',
    placeholder: 'Enter text...',
    description: 'The plaintext message to encrypt or ciphertext to decrypt.',
  },
  {
    name: 'keyword',
    label: 'Keyword',
    type: 'text',
    placeholder: 'Enter keyword...',
    description: 'The keyword used to generate the Polybius square.',
  },
];

export default function BifidPage() {
  return (
    <CipherToolPage
      title="Bifid Cipher"
      description="A fractionation cipher combining the Polybius square with transposition. Invented by Felix Delastelle in 1901, it is notable because it breaks characters into smaller parts, making it resistant to simple frequency analysis."
      icon={Grid3X3}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/classical/bifid/encrypt"
      decryptEndpoint="/classical/bifid/decrypt"
    />
  );
}
