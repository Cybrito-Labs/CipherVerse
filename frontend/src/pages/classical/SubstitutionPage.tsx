import { z } from 'zod';
import { Sigma } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
  keyword: z.string().min(26, 'Keyword must be a 26-character alphabet permutation').max(26),
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
    label: 'Substitution Alphabet',
    type: 'text',
    placeholder: 'QWERTYUIOPASDFGHJKLZXCVBNM',
    description: 'A 26-character string representing the new alphabet order.',
  },
];

export default function SubstitutionPage() {
  return (
    <CipherToolPage
      title="Substitution Cipher"
      description="A method of encoding by which units of plaintext are replaced with ciphertext, according to a fixed system. The 'units' may be single letters, pairs of letters, triplets of letters, mixtures of the above, and so forth."
      icon={Sigma}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/classical/substitute/encrypt"
      decryptEndpoint="/classical/substitute/decrypt"
    />
  );
}
