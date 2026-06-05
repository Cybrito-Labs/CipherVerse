import { z } from 'zod';
import { Image as ImageIcon } from 'lucide-react';
import CipherToolPage from '@/components/shared/CipherToolPage';
import type { FieldConfig } from '@/components/shared/CipherToolPage';

const schema = z.object({
  input_img: z.string().min(1, 'Input path is required'),
  output_img: z.string().min(1, 'Output path is required'),
  secret: z.string().min(1, 'Secret message is required'),
});

const fields: FieldConfig[] = [
  {
    name: 'input_img',
    label: 'Input Image Path',
    type: 'text',
    placeholder: 'e.g., C:/Images/cover.png',
    description: 'Absolute path to the source image file on the backend filesystem.',
  },
  {
    name: 'output_img',
    label: 'Output Image Path',
    type: 'text',
    placeholder: 'e.g., C:/Images/stego.png',
    description: 'Absolute path where the stego image will be saved (must be PNG).',
  },
  {
    name: 'secret',
    label: 'Secret Message',
    type: 'textarea',
    placeholder: 'Enter the secret data to embed...',
    description: 'The secret text to hide inside the image using LSB encoding.',
  },
];

export default function ImageStegoPage() {
  return (
    <CipherToolPage
      title="Image Steganography"
      description="Embed secret data into an image file using Least Significant Bit (LSB) encoding. The resulting image will look identical to the original to the human eye."
      icon={ImageIcon}
      fields={fields}
      endpoint="/steganography/image/encode"
      schema={schema}
      hasTabs={false}
    />
  );
}
