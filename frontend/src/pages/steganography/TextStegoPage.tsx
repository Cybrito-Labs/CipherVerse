import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { MessageSquare } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import {
  ToolPageLayout,
  ToolInputPanel,
  ToolResultPanel,
  ToolTabs,
  ToolActions
} from '@/components/shared/layout';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';

const encSchema = z.object({
  cover_text: z.string().min(1, 'Cover text is required'),
  secret: z.string().min(1, 'Secret is required'),
});

const decSchema = z.object({
  text: z.string().min(1, 'Stego text is required'),
});

interface StegoResponse {
  result: string;
}

export default function TextStegoPage() {
  const [activeTab, setActiveTab] = useState('encode');

  const encMutation = useMutation<StegoResponse, Error, z.infer<typeof encSchema>>({
    mutationFn: async (data) => {
      return await api.post<StegoResponse>('/steganography/text/encode', data);
    },
  });

  const decMutation = useMutation<StegoResponse, Error, z.infer<typeof decSchema>>({
    mutationFn: async (data) => {
      return await api.post<StegoResponse>('/steganography/text/decode', data);
    },
  });

  const encForm = useForm<z.infer<typeof encSchema>>({ resolver: zodResolver(encSchema) });
  const decForm = useForm<z.infer<typeof decSchema>>({ resolver: zodResolver(decSchema) });

  const handleClear = () => {
    encForm.reset();
    decForm.reset();
    encMutation.reset();
    decMutation.reset();
  };

  const handleExecute = () => {
    if (activeTab === 'encode') encForm.handleSubmit((d) => encMutation.mutate(d))();
    else decForm.handleSubmit((d) => decMutation.mutate(d))();
  };

  return (
    <ToolPageLayout
      title="Text Steganography"
      description="Hide secret messages within seemingly normal cover text. The secret is encoded using invisible zero-width characters."
      icon={MessageSquare}
      badges={[{ label: 'Steganography', variant: 'default' }]}
    >
      <ToolInputPanel>
        <ToolTabs
          tabs={[
            { id: 'encode', label: 'Encode Secret' },
            { id: 'decode', label: 'Decode Secret' },
          ]}
          activeTab={activeTab}
          onTabChange={(v) => { setActiveTab(v); encMutation.reset(); decMutation.reset(); }}
        />

        {activeTab === 'encode' && (
          <form className="space-y-6">
            <div className="space-y-3">
              <Label className="text-foreground">Cover Text (Normal visible text)</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground min-h-[100px]" placeholder="e.g., Hello World" {...encForm.register('cover_text')} />
              {encForm.formState.errors.cover_text && <p className="text-sm text-destructive">{encForm.formState.errors.cover_text.message}</p>}
            </div>
            <div className="space-y-3">
              <Label className="text-foreground">Secret Message (Hidden)</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground min-h-[100px]" placeholder="e.g., Attack at dawn" {...encForm.register('secret')} />
              {encForm.formState.errors.secret && <p className="text-sm text-destructive">{encForm.formState.errors.secret.message}</p>}
            </div>
          </form>
        )}

        {activeTab === 'decode' && (
          <form className="space-y-6">
            <div className="space-y-3">
              <Label className="text-foreground">Stego Text (Text containing zero-width characters)</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground h-[180px]" placeholder="Paste the text containing the hidden message here..." {...decForm.register('text')} />
              {decForm.formState.errors.text && <p className="text-sm text-destructive">{decForm.formState.errors.text.message}</p>}
            </div>
          </form>
        )}

        <ToolActions
          isExecuting={encMutation.isPending || decMutation.isPending}
          onExecute={handleExecute}
          onClear={handleClear}
          executeLabel={activeTab === 'encode' ? 'Encode' : 'Decode'}
        />
      </ToolInputPanel>

      <ToolResultPanel
        title={activeTab === 'encode' ? 'Stego Text Result' : 'Extracted Secret Message'}
        result={activeTab === 'encode' ? encMutation.data?.result : decMutation.data?.result}
        isLoading={encMutation.isPending || decMutation.isPending}
        error={activeTab === 'encode' ? encMutation.error : decMutation.error}
        onRetry={handleExecute}
        onClear={handleClear}
        emptyMessage={activeTab === 'encode' ? 'Provide cover text and a secret to embed using zero-width characters.' : 'Paste stego text to extract the hidden message.'}
      />
    </ToolPageLayout>
  );
}
