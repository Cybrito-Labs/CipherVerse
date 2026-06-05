import { z } from 'zod';
import { Shuffle } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
});

const fields: FieldConfig[] = [
  {
    name: 'text',
    label: 'Text',
    type: 'textarea',
    placeholder: 'Enter text...',
    description: 'The Atbash cipher reverses the alphabet: A↔Z, B↔Y, C↔X, etc. It is its own inverse.',
  },
];

export default function AtbashPage() {
  return (
    <CipherToolPage
      title="Atbash Cipher"
      description="A simple substitution cipher originally used for the Hebrew alphabet. Each letter is mapped to its reverse: A becomes Z, B becomes Y, and so on. Since the cipher is symmetric, encoding and decoding are the same operation."
      icon={Shuffle}
      fields={fields}
      endpoint="/classical/atbash"
      schema={schema}
    />
  );
}
