import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, Binary } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
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
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 lg:col-span-4 h-fit">
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
                <Play className="w-4 h-4"/> Run Tests
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

        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="lg:col-span-8">
          {!res && !mutation.isPending && !mutation.error ? (
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center min-h-[400px]">
              <Binary className="w-16 h-16 mb-4 opacity-20" />
              <p>Enter a file path to perform randomness evaluations.</p>
            </div>
          ) : (
            <ResultPanel
              title="Statistical Randomness Report"
              isLoading={mutation.isPending}
              error={mutation.error}
            >
              {res && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <MetricsPanel title="Core Metrics">
                      <MetricRow label="Shannon Entropy" value={res.entropy.toFixed(4)} />
                      <MetricRow label="Chi-Square (χ²)" value={res.chi_square.toFixed(2)} />
                      <MetricRow label="Total Runs" value={res.runs.toLocaleString()} />
                    </MetricsPanel>

                    <div className="glass rounded-xl p-4 flex flex-col items-center justify-center border border-border">
                       <h4 className="text-sm font-semibold text-primary uppercase tracking-wider self-start mb-2">Verdict (Estimates)</h4>
                       <div className="text-center">
                          {res.entropy > 7.9 && res.chi_square < 300 ? (
                            <span className="text-success font-bold text-xl">Cryptographically Random</span>
                          ) : res.entropy > 7.0 ? (
                            <span className="text-warning font-bold text-xl">Likely Packed/Encrypted</span>
                          ) : (
                            <span className="text-destructive font-bold text-xl">Highly Non-Random</span>
                          )}
                       </div>
                    </div>
                  </div>

                  <div className="glass rounded-xl p-6 border border-border">
                    <h4 className="text-sm font-semibold text-primary uppercase tracking-wider mb-6">Bit Distribution (0s vs 1s)</h4>
                    <div className="h-[200px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff1a" vertical={false} />
                          <XAxis dataKey="name" tick={{fill: '#888888'}} axisLine={false} tickLine={false} />
                          <YAxis domain={[0, 100]} tick={{fill: '#888888'}} axisLine={false} tickLine={false} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#111111', borderColor: '#333', borderRadius: '8px' }}
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
            </ResultPanel>
          )}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
