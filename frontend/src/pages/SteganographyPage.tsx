import { Image as ImageIcon, MessageSquare, Music } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ToolCard } from '@/components/shared/ToolCard';

const tools = [
  { title: 'Text Steganography', description: 'Hide secret messages within seemingly innocent text using zero-width characters.', icon: MessageSquare, path: '/steganography/text' },
  { title: 'Image Steganography', description: 'Embed and extract data from images using Least Significant Bit (LSB) encoding.', icon: ImageIcon, path: '/steganography/image' },
  { title: 'Audio Steganography', description: 'Hide data within WAV audio files by manipulating the lowest bits of audio samples.', icon: Music, path: '/steganography/audio' },
];

export default function SteganographyPage() {
  return (
    <ToolPageLayout
      title="Steganography Tools"
      description="Steganography is the practice of concealing a file, message, image, or video within another file, message, image, or video."
      icon={ImageIcon}
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {tools.map((tool) => (
          <ToolCard
            key={tool.path}
            title={tool.title}
            description={tool.description}
            icon={tool.icon}
            path={tool.path}
          />
        ))}
      </div>
    </ToolPageLayout>
  );
}
