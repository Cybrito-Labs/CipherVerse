import { z } from 'zod';
import { BookOpen } from 'lucide-react';
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
    description: 'The plaintext message to encode or Bacon-encoded string to decode.',
  },
];

export default function BaconPage() {
  return (
    <CipherToolPage
      title="Bacon Cipher"
      description="A steganographic method of message encoding developed by Sir Francis Bacon in 1605. It replaces each letter with a 5-letter sequence of 'A' and 'B', which can be hidden in the formatting of normal text."
      icon={BookOpen}
      fields={fields}
      endpoint=""
      schema={schema}
      hasTabs
      encryptEndpoint="/classical/bacon/encrypt"
      decryptEndpoint="/classical/bacon/decrypt"
    />
  );
}
