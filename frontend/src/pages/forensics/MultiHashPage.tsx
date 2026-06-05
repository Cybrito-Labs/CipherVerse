import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, Search } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToolExecution } from '@/hooks/useToolExecution';
import { MetricsPanel, MetricRow } from '@/components/dashboard/MetricsPanel';

const schema = z.object({
  filepath: z.string().min(1, 'File path is required'),
});

interface MultiHashResponse {
  hashes: Record<string, string>;
}

export default function MultiHashPage() {
  const mutation = useToolExecution<z.infer<typeof schema>, MultiHashResponse>({ endpoint: '/file-tools/multi-hash' });
  const form = useForm<z.infer<typeof schema>>({ resolver: zodResolver(schema), defaultValues: { filepath: '' } });

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  return (
    <ToolPageLayout
      title="Multi Hash Calculator"
      description="Simultaneously calculate MD5, SHA-1, SHA-256, and SHA-512 hashes for comprehensive file identification and VirusTotal lookups."
      icon={Search}
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 lg:col-span-5 h-fit">
          <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-4">
            <div className="space-y-2">
              <Label>Target Filepath</Label>
              <Input 
                placeholder="C:/evidence/suspect.exe" 
                className="bg-background/50 font-mono" 
                {...form.register('filepath')} 
              />
              {form.formState.errors.filepath && <p className="text-xs text-destructive">{form.formState.errors.filepath.message}</p>}
            </div>
            
            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> Calculate All Hashes
              </Button>
              <Button type="button" variant="outline" onClick={handleClear} className="px-3">
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>
            {mutation.error && (
              <div className="p-3 mt-4 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-lg">
                {mutation.error.message}
              </div>
            )}
          </form>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="lg:col-span-7">
          {!res && !mutation.isPending && !mutation.error ? (
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center min-h-[400px]">
              <Search className="w-16 h-16 mb-4 opacity-20" />
              <p>Enter a file path to calculate all standard hashes at once.</p>
            </div>
          ) : (
            <ResultPanel
              title="Hash Digest Report"
              isLoading={mutation.isPending}
              error={mutation.error}
            >
              {res && (
                <MetricsPanel title="Computed Cryptographic Hashes">
                  {Object.entries(res.hashes).map(([algo, hash]) => (
                    <MetricRow key={algo} label={algo} value={hash} isMono />
                  ))}
                </MetricsPanel>
              )}
            </ResultPanel>
          )}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
