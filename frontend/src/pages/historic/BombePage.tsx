import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Search } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import {
  ToolPageLayout,
  ToolInputPanel,
  ToolResultPanel,
  ToolActions
} from '@/components/shared/layout';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

const schema = z.object({
  ciphertext: z.string().min(1, 'Ciphertext is required').regex(/^[a-zA-Z\s]+$/, "Only letters and spaces allowed"),
  crib: z.string().min(1, 'Crib is required').regex(/^[a-zA-Z\s]+$/, "Only letters and spaces allowed"),
  r1_type: z.string(),
  r2_type: z.string(),
  r3_type: z.string(),
});

interface BombeResponse {
  result: string;
  matches: string[];
}

export default function BombePage() {
  const mutation = useMutation<BombeResponse, Error, z.infer<typeof schema>>({
    mutationFn: async (data) => {
      const res = await api.post('/historic/bombe', {
        ciphertext: data.ciphertext,
        crib: data.crib,
        rotor_order: [data.r1_type, data.r2_type, data.r3_type],
      });
      return res.data;
    },
  });

  const form = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
    defaultValues: {
      ciphertext: '',
      crib: '',
      r1_type: 'I',
      r2_type: 'II',
      r3_type: 'III'
    }
  });

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  return (
    <ToolPageLayout
      title="Turing Bombe Simulator"
      description="Simulate Alan Turing's Bombe to cryptanalyze Enigma. Provide the intercepted ciphertext and a 'crib' (a known plaintext guess like 'WETTERBERICHT') to brute-force the possible rotor starting positions."
      icon={Search}
      badges={[{ label: 'Historical', variant: 'warning' }, { label: 'Cryptanalysis', variant: 'default' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="grid grid-cols-3 gap-2">
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Left Rotor</Label>
              <Select onValueChange={(val) => form.setValue('r1_type', val)} defaultValue={form.getValues('r1_type')}>
                <SelectTrigger className="bg-background border-border text-foreground"><SelectValue /></SelectTrigger>
                <SelectContent className="bg-card border-border">
                  {['I','II','III','IV','V'].map(r => <SelectItem key={r} value={r} className="text-foreground hover:bg-secondary focus:bg-secondary">{r}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Mid Rotor</Label>
              <Select onValueChange={(val) => form.setValue('r2_type', val)} defaultValue={form.getValues('r2_type')}>
                <SelectTrigger className="bg-background border-border text-foreground"><SelectValue /></SelectTrigger>
                <SelectContent className="bg-card border-border">
                  {['I','II','III','IV','V'].map(r => <SelectItem key={r} value={r} className="text-foreground hover:bg-secondary focus:bg-secondary">{r}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Right Rotor</Label>
              <Select onValueChange={(val) => form.setValue('r3_type', val)} defaultValue={form.getValues('r3_type')}>
                <SelectTrigger className="bg-background border-border text-foreground"><SelectValue /></SelectTrigger>
                <SelectContent className="bg-card border-border">
                  {['I','II','III','IV','V'].map(r => <SelectItem key={r} value={r} className="text-foreground hover:bg-secondary focus:bg-secondary">{r}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-3">
            <Label className="text-foreground">Intercepted Ciphertext</Label>
            <Textarea
              placeholder="Enter Enigma encrypted text..."
              className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground font-mono min-h-[100px] uppercase"
              {...form.register('ciphertext')}
              onChange={(e) => e.target.value = e.target.value.replace(/[^a-zA-Z\s]/g, '').toUpperCase()}
            />
            {form.formState.errors.ciphertext && <p className="text-sm text-destructive">{form.formState.errors.ciphertext.message}</p>}
          </div>

          <div className="space-y-3">
            <Label className="text-foreground">Plaintext Crib Guess</Label>
            <Input
              placeholder="e.g., WETTER"
              className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground font-mono uppercase"
              {...form.register('crib')}
              onChange={(e) => e.target.value = e.target.value.replace(/[^a-zA-Z\s]/g, '').toUpperCase()}
            />
            <p className="text-[11px] text-muted-foreground">The crib must be equal to or shorter than the ciphertext.</p>
            {form.formState.errors.crib && <p className="text-sm text-destructive">{form.formState.errors.crib.message}</p>}
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Run Bombe Analysis"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Analysis Output"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Configure your crib and run the analysis."
      >
        {res && (
          <div className="space-y-4 pt-2">
            <div className="p-4 rounded-xl border border-primary/20 bg-primary/5 text-primary text-sm font-semibold">
              {res.result}
            </div>

            <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mt-4">Potential Initial Settings (Matches)</h4>
            {res.matches.length === 0 ? (
              <div className="p-8 text-center text-muted-foreground border border-dashed border-border rounded-xl">
                No matching rotor positions found for this crib.
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-[400px] overflow-y-auto pr-2">
                {res.matches.map((match, i) => (
                  <div key={i} className="p-3 bg-background border border-border rounded-lg text-center font-mono font-bold tracking-widest text-foreground">
                    {match}
                  </div>
                ))}
              </div>
            )}

            {res.matches.length > 0 && (
              <p className="text-xs text-muted-foreground mt-4">
                Take these 3-letter settings and test them in the Enigma Machine tool to see if the full message decrypts into readable German.
              </p>
            )}
          </div>
        )}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
