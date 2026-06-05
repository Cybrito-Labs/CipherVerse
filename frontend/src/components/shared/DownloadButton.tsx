import { useState } from 'react';
import { Download, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface DownloadButtonProps {
  content: string;
  filename: string;
  label?: string;
  className?: string;
}

export function DownloadButton({ content, filename, label = 'Download', className }: DownloadButtonProps) {
  const [downloaded, setDownloaded] = useState(false);

  const handleDownload = () => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    setDownloaded(true);
    setTimeout(() => setDownloaded(false), 2000);
  };

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={handleDownload}
      className={className}
    >
      {downloaded ? <Check className="w-4 h-4 mr-1.5 text-success" /> : <Download className="w-4 h-4 mr-1.5" />}
      {downloaded ? 'Downloaded' : label}
    </Button>
  );
}
