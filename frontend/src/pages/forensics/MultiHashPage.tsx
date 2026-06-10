import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Search } from 'lucide-react';
import {
  ToolPageLayout,
  ToolInputPanel,
  ToolResultPanel,
  ToolActions
} from '@/components/shared/layout';
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
      badges={[{ label: 'Forensics', variant: 'default' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="space-y-3">
            <Label className="text-[#EDEDED]">Target Filepath</Label>
            <Input
              placeholder="C:/evidence/suspect.exe"
              className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] placeholder:text-[#52525B] font-mono"
              {...form.register('filepath')}
            />
            {form.formState.errors.filepath && <p className="text-sm text-destructive">{form.formState.errors.filepath.message}</p>}
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Calculate All Hashes"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Hash Digest Report"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Enter a file path to calculate all standard hashes at once."
      >
        {res && (
          <MetricsPanel title="Computed Cryptographic Hashes">
            {Object.entries(res.hashes).map(([algo, hash]) => (
              <MetricRow key={algo} label={algo} value={hash} isMono />
            ))}
          </MetricsPanel>
        )}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
