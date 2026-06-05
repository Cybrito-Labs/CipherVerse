import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, Activity, AlertTriangle } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToolExecution } from '@/hooks/useToolExecution';
import { RiskCard } from '@/components/dashboard/RiskCard';
import { ProgressBar } from '@/components/dashboard/ProgressBar';

const schema = z.object({
  filepath: z.string().min(1, 'File path is required'),
});

interface EntropyResponse {
  entropy: number;
  interpretation: string;
}

export default function EntropyPage() {
  const mutation = useToolExecution<z.infer<typeof schema>, EntropyResponse>({ endpoint: '/file-tools/entropy' });
  const form = useForm<z.infer<typeof schema>>({ resolver: zodResolver(schema), defaultValues: { filepath: '' } });

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  const getRiskLevel = (entropy: number) => {
    if (entropy > 7.5) return 'high';
    if (entropy > 6.5) return 'medium';
    return 'low';
  };

  const risk = res ? getRiskLevel(res.entropy) : 'neutral';

  return (
    <ToolPageLayout
      title="Shannon Entropy"
      description="Calculate the Shannon Entropy of a file (scale 0.0 to 8.0). High entropy values (> 7.2) strongly indicate that the file is compressed, packed, or encrypted."
      icon={Activity}
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
                <Play className="w-4 h-4"/> Calculate Entropy
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
              <Activity className="w-16 h-16 mb-4 opacity-20" />
              <p>Enter a file path to calculate its overall entropy.</p>
            </div>
          ) : (
            <ResultPanel
              title="Entropy Report"
              isLoading={mutation.isPending}
              error={mutation.error}
            >
              {res && (
                <div className="space-y-8">
                  <RiskCard
                    title="Shannon Entropy Score"
                    value={res.entropy.toFixed(3)}
                    description={res.interpretation}
                    icon={AlertTriangle}
                    riskLevel={risk}
                  />

                  <div className="glass rounded-xl p-6">
                    <h4 className="text-sm font-semibold text-primary uppercase tracking-wider mb-4">Entropy Scale</h4>
                    
                    <div className="flex justify-between text-xs text-muted-foreground mb-2">
                      <span>0.0 (Empty)</span>
                      <span>Text (~4-5)</span>
                      <span>Native Exec (~6-6.5)</span>
                      <span className="text-destructive font-bold">Packed (7.5 - 8.0)</span>
                    </div>
                    
                    <ProgressBar 
                      value={res.entropy} 
                      max={8.0} 
                      height="lg" 
                      colorClass={risk === 'high' ? 'bg-destructive' : risk === 'medium' ? 'bg-warning' : 'bg-success'} 
                    />
                  </div>
                </div>
              )}
            </ResultPanel>
          )}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
