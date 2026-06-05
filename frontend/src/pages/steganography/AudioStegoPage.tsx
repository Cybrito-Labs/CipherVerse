import { z } from 'zod';
import { Music } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  input_wav: z.string().min(1, 'Input path is required'),
  output_wav: z.string().min(1, 'Output path is required'),
  secret: z.string().min(1, 'Secret message is required'),
});

const fields: FieldConfig[] = [
  {
    name: 'input_wav',
    label: 'Input Audio Path (WAV)',
    type: 'text',
    placeholder: 'e.g., C:/Audio/cover.wav',
    description: 'Absolute path to the source WAV file on the backend filesystem.',
  },
  {
    name: 'output_wav',
    label: 'Output Audio Path (WAV)',
    type: 'text',
    placeholder: 'e.g., C:/Audio/stego.wav',
    description: 'Absolute path where the stego audio will be saved.',
  },
  {
    name: 'secret',
    label: 'Secret Message',
    type: 'textarea',
    placeholder: 'Enter the secret data to embed...',
    description: 'The secret text to hide inside the audio using LSB encoding.',
  },
];

export default function AudioStegoPage() {
  return (
    <CipherToolPage
      title="Audio Steganography"
      description="Hide secret messages within WAV audio files by manipulating the lowest bits of audio samples. The audio quality remains perceptually unchanged."
      icon={Music}
      fields={fields}
      endpoint="/steganography/audio/encode"
      schema={schema}
      hasTabs={false}
    />
  );
}
