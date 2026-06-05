import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, MessageSquare } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

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
      const res = await api.post('/steganography/text/encode', data);
      return res.data;
    },
  });

  const decMutation = useMutation<StegoResponse, Error, z.infer<typeof decSchema>>({
    mutationFn: async (data) => {
      // Decode uses query parameter as per backend design
      const res = await api.post(`/steganography/text/decode?text=${encodeURIComponent(data.text)}`);
      return res.data;
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

  return (
    <ToolPageLayout
      title="Text Steganography"
      description="Hide secret messages within seemingly normal cover text. The secret is encoded using invisible zero-width characters."
      icon={MessageSquare}
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 h-fit">
          <Tabs value={activeTab} onValueChange={(v) => { setActiveTab(v); encMutation.reset(); decMutation.reset(); }}>
            <TabsList className="mb-4 bg-background/50 grid w-full grid-cols-2">
              <TabsTrigger value="encode">Encode Secret</TabsTrigger>
              <TabsTrigger value="decode">Decode Secret</TabsTrigger>
            </TabsList>

            <TabsContent value="encode">
              <form onSubmit={encForm.handleSubmit((d) => encMutation.mutate(d))} className="space-y-4">
                <div className="space-y-2">
                  <Label>Cover Text (Normal visible text)</Label>
                  <Textarea className="bg-background/50" placeholder="e.g., Hello World" {...encForm.register('cover_text')} />
                  {encForm.formState.errors.cover_text && <p className="text-xs text-destructive">{encForm.formState.errors.cover_text.message}</p>}
                </div>
                <div className="space-y-2">
                  <Label>Secret Message (Hidden)</Label>
                  <Textarea className="bg-background/50" placeholder="e.g., Attack at dawn" {...encForm.register('secret')} />
                  {encForm.formState.errors.secret && <p className="text-xs text-destructive">{encForm.formState.errors.secret.message}</p>}
                </div>
                <div className="flex gap-2 pt-2">
                  <Button type="submit" disabled={encMutation.isPending} className="w-full gap-2"><Play className="w-4 h-4"/> Encode</Button>
                  <Button type="button" variant="outline" onClick={handleClear} className="px-3"><RotateCcw className="w-4 h-4" /></Button>
                </div>
              </form>
            </TabsContent>

            <TabsContent value="decode">
              <form onSubmit={decForm.handleSubmit((d) => decMutation.mutate(d))} className="space-y-4">
                <div className="space-y-2">
                  <Label>Stego Text (Text containing zero-width characters)</Label>
                  <Textarea className="bg-background/50 h-[180px]" placeholder="Paste the text containing the hidden message here..." {...decForm.register('text')} />
                  {decForm.formState.errors.text && <p className="text-xs text-destructive">{decForm.formState.errors.text.message}</p>}
                </div>
                <div className="flex gap-2 pt-2">
                  <Button type="submit" disabled={decMutation.isPending} className="w-full gap-2"><Play className="w-4 h-4"/> Decode</Button>
                  <Button type="button" variant="outline" onClick={handleClear} className="px-3"><RotateCcw className="w-4 h-4" /></Button>
                </div>
              </form>
            </TabsContent>
          </Tabs>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
          <ResultPanel
            title={activeTab === 'encode' ? 'Stego Text Result' : 'Extracted Secret Message'}
            result={activeTab === 'encode' ? encMutation.data?.result : decMutation.data?.result}
            isLoading={encMutation.isPending || decMutation.isPending}
            error={activeTab === 'encode' ? encMutation.error : decMutation.error}
          />
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
