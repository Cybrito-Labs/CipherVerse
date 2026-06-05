import { z } from 'zod';
import { RotateCcw } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  text: z.string().min(1, 'Text is required'),
  shift: z.number().int().min(0).max(25),
});

const fields: FieldConfig[] = [
  {
    name: 'text',
    label: 'Text',
    type: 'textarea',
    placeholder: 'Enter text to encrypt...',
    description: 'The plaintext message to encrypt using the Caesar cipher.',
  },
  {
    name: 'shift',
    label: 'Shift',
    type: 'number',
    placeholder: '3',
    defaultValue: 3,
    description: 'Number of positions to shift each letter (0–25).',
  },
];

export default function CaesarPage() {
  return (
    <CipherToolPage
      title="Caesar Cipher"
      description="One of the earliest known ciphers. Each letter is shifted by a fixed number of positions in the alphabet. Named after Julius Caesar who used it in his private correspondence."
      icon={RotateCcw}
      fields={fields}
      endpoint="/classical/caesar"
      schema={schema}
    />
  );
}
