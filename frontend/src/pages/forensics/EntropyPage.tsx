import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Activity, AlertTriangle } from 'lucide-react';
import {
  ToolPageLayout,
  ToolInputPanel,
  ToolResultPanel,
  ToolActions
} from '@/components/shared/layout';
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
      badges={[{ label: 'Forensics', variant: 'default' }, { label: 'Analysis', variant: 'warning' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="space-y-3">
            <Label className="text-foreground">Target Filepath</Label>
            <Input
              placeholder="C:/evidence/suspect.exe"
              className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground font-mono"
              {...form.register('filepath')}
            />
            {form.formState.errors.filepath && <p className="text-sm text-destructive">{form.formState.errors.filepath.message}</p>}
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Calculate Entropy"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Entropy Report"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Enter a file path to calculate its overall entropy."
      >
        {res && (
          <div className="space-y-8 pt-2">
            <RiskCard
              title="Shannon Entropy Score"
              value={res.entropy.toFixed(3)}
              description={res.interpretation}
              icon={AlertTriangle}
              riskLevel={risk}
            />

            <div className="bg-background border border-border rounded-[14px] p-6 shadow-sm flex flex-col gap-6">
              <h4 className="text-sm font-semibold text-primary uppercase tracking-wider mb-4">Entropy Scale</h4>

              <div className="flex justify-between text-xs text-muted-foreground mb-2">
                <span>0.0 (Empty)</span>
                <span>Text (~4-5)</span>
                <span>Native Exec (~6-6.5)</span>
                <span className="text-[#F87171] font-bold">Packed (7.5 - 8.0)</span>
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
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
