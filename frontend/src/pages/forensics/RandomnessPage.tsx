import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Binary } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
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

interface RandomnessResponse {
  entropy: number;
  bit_balance: Record<string, number>;
  runs: number;
  chi_square: number;
}

export default function RandomnessPage() {
  const mutation = useToolExecution<z.infer<typeof schema>, RandomnessResponse>({ endpoint: '/file-tools/randomness-test' });
  const form = useForm<z.infer<typeof schema>>({ resolver: zodResolver(schema), defaultValues: { filepath: '' } });

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  const chartData = res ? Object.entries(res.bit_balance).map(([key, value]) => ({
    name: `Bit ${key}`,
    value: value
  })) : [];

  return (
    <ToolPageLayout
      title="Randomness Test"
      description="Perform statistical tests (Entropy, Chi-Square, Bit Balance, Runs Test) to evaluate the cryptographic strength and true randomness of a file's contents."
      icon={Binary}
      badges={[{ label: 'Forensics', variant: 'default' }, { label: 'Statistical', variant: 'success' }]}
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
            executeLabel="Run Tests"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Statistical Randomness Report"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Enter a file path to perform randomness evaluations."
      >
        {res && (
          <div className="space-y-6 pt-2">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <MetricsPanel title="Core Metrics">
                <MetricRow label="Shannon Entropy" value={res.entropy.toFixed(4)} />
                <MetricRow label="Chi-Square (χ²)" value={res.chi_square.toFixed(2)} />
                <MetricRow label="Total Runs" value={res.runs.toLocaleString()} />
              </MetricsPanel>

              <div className="bg-background border border-border rounded-xl p-4 flex flex-col items-center justify-center">
                 <h4 className="text-sm font-semibold text-primary uppercase tracking-wider self-start mb-2">Verdict (Estimates)</h4>
                 <div className="text-center">
                    {res.entropy > 7.9 && res.chi_square < 300 ? (
                      <span className="text-[#4ADE80] font-bold text-xl">Cryptographically Random</span>
                    ) : res.entropy > 7.0 ? (
                      <span className="text-[#FBBF24] font-bold text-xl">Likely Packed/Encrypted</span>
                    ) : (
                      <span className="text-[#F87171] font-bold text-xl">Highly Non-Random</span>
                    )}
                 </div>
              </div>
            </div>

            <div className="bg-background border border-border rounded-xl p-6">
              <h4 className="text-sm font-semibold text-primary uppercase tracking-wider mb-6">Bit Distribution (0s vs 1s)</h4>
              <div className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff1a" vertical={false} />
                    <XAxis dataKey="name" tick={{fill: '#A1A1AA'}} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 100]} tick={{fill: '#A1A1AA'}} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0A0A0A', borderColor: '#27272A', borderRadius: '8px' }}
                      itemStyle={{ color: '#00e5ff' }}
                      formatter={(val: number) => [`${val.toFixed(2)}%`, 'Frequency']}
                    />
                    <Bar dataKey="value" fill="#00e5ff" radius={[4, 4, 0, 0]} maxBarSize={60} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-center text-muted-foreground mt-4">
                Perfect randomness approaches exactly 50.00% for both bits.
              </p>
            </div>
          </div>
        )}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
