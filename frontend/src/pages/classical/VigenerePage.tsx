import { z } from 'zod';
import { KeyRound } from 'lucide-react';
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
    description: 'The message to encrypt or decrypt.',
  },
  {
    name: 'keyword',
    label: 'Keyword',
    type: 'text',
    placeholder: 'Enter keyword...',
    description: 'The keyword used for polyalphabetic substitution.',
  },
];

export default function VigenerePage() {
  return (
    <CipherToolPage
      title="Vigenère Cipher"
      description="A polyalphabetic substitution cipher that uses a keyword to shift each letter by a different amount. Invented by Blaise de Vigenère in the 16th century and was considered unbreakable for 300 years."
      icon={KeyRound}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/classical/vigenere/encrypt"
      decryptEndpoint="/classical/vigenere/decrypt"
    />
  );
}
