import { useParams, Navigate } from 'react-router-dom';
import { z } from 'zod';
import { FileCode, Hash, Link as LinkIcon, Binary, Code } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  data: z.string().min(1, 'Input data is required'),
});

const fields: FieldConfig[] = [
  {
    name: 'data',
    label: 'Input Data',
    type: 'textarea',
    placeholder: 'Enter data...',
    description: 'The data to encode or decode.',
  },
];

const toolConfigs: Record<string, { title: string; description: string; icon: any; endpoint: string }> = {
  base64: {
    title: 'Base64',
    description: 'Encode and decode Base64 strings. Widely used for encoding binary data as text.',
    icon: FileCode,
    endpoint: '/encoding/base64',
  },
  hex: {
    title: 'Hexadecimal',
    description: 'Convert text to and from hexadecimal representation.',
    icon: Hash,
    endpoint: '/encoding/hex',
  },
  url: {
    title: 'URL Encoding',
    description: 'Percent-encoding for safely transmitting data in URLs.',
    icon: LinkIcon,
    endpoint: '/encoding/url',
  },
  binary: {
    title: 'Binary Encoding',
    description: 'Convert text to binary (base-2) and vice versa.',
    icon: Binary,
    endpoint: '/encoding/binary',
  },
  morse: {
    title: 'Morse Code',
    description: 'Translate text to standard Morse code dots and dashes.',
    icon: Code,
    endpoint: '/encoding/morse',
  },
};

export default function EncodingToolPage() {
  const { tool } = useParams<{ tool: string }>();

  if (!tool || !toolConfigs[tool]) {
    return <Navigate to="/encoding" replace />;
  }

  const config = toolConfigs[tool];

  return (
    <CipherToolPage
      title={config.title}
      description={config.description}
      icon={config.icon}
      fields={fields}
      endpoint="" // Not used since hasTabs is true
      schema={schema}
      hasTabs
      encryptEndpoint={`${config.endpoint}/encode`}
      decryptEndpoint={`${config.endpoint}/decode`}
    />
  );
}
