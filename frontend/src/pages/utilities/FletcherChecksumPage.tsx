import { z } from 'zod';
import { FileCode2 } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  data: z.string().min(1, 'Data is required'),
});

const fields: FieldConfig[] = [
  {
    name: 'data',
    label: 'Data string',
    type: 'textarea',
    placeholder: 'Enter the text you want to checksum...',
    description: 'The plaintext data to calculate the 16-bit Fletcher checksum for.',
  },
];

export default function FletcherChecksumPage() {
  return (
    <CipherToolPage
      title="Fletcher-16 Checksum"
      description="The Fletcher checksum is an algorithm for computing a position-dependent checksum. It is designed to provide error-detection properties approaching those of a cyclic redundancy check (CRC) but with lower computational effort."
      icon={FileCode2}
      fields={fields}
      endpoint="/utilities/checksum/fletcher16"
      schema={schema}
      hasTabs={false}
    />
  );
}
