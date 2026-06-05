import { z } from 'zod';
import { Fingerprint } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  data: z.string().min(1, 'Data is required'),
  algorithm: z.enum(['sha256', 'sha1', 'md5']),
});

const fields: FieldConfig[] = [
  {
    name: 'data',
    label: 'Data / Certificate (PEM or raw string)',
    type: 'textarea',
    placeholder: 'Enter data to fingerprint...',
    description: 'The raw string or PEM encoded certificate to hash.',
  },
  {
    name: 'algorithm',
    label: 'Hash Algorithm',
    type: 'select',
    placeholder: 'Select algorithm...',
    defaultValue: 'sha256',
    description: 'The hashing algorithm used to generate the fingerprint.',
    isAdvanced: true,
    options: [
      { label: 'SHA-256 (Recommended)', value: 'sha256' },
      { label: 'SHA-1 (Legacy)', value: 'sha1' },
      { label: 'MD5 (Insecure)', value: 'md5' },
    ],
  },
];

export default function FingerprintPage() {
  return (
    <CipherToolPage
      title="Fingerprint Generator"
      description="Generate cryptographic fingerprints (hashes) from raw data or certificates. Fingerprints are widely used to verify the integrity and authenticity of public keys and certificates."
      icon={Fingerprint}
      fields={fields}
      endpoint="/certificates/x509/fingerprint"
      schema={schema}
      hasTabs={false}
      transformPayload={(data) => {
        return {
          data: data.data,
          algorithm: data.algorithm,
        };
      }}
    />
  );
}
