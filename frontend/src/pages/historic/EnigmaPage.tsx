import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { RotateCw } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import {
  ToolPageLayout,
  ToolInputPanel,
  ToolResultPanel,
  ToolActions
} from '@/components/shared/layout';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { RotorControl } from '@/components/shared/RotorControl';

const schema = z.object({
  text: z.string().min(1, 'Text is required').regex(/^[a-zA-Z\s]+$/, "Only letters and spaces allowed"),
  r1_type: z.string(),
  r2_type: z.string(),
  r3_type: z.string(),
});

interface EnigmaResponse {
  result: string;
}

export default function EnigmaPage() {
  const [positions, setPositions] = useState<[number, number, number]>([0, 0, 0]);

  const mutation = useMutation<EnigmaResponse, Error, z.infer<typeof schema>>({
    mutationFn: async (data) => {
      const res = await api.post('/historic/enigma', {
        text: data.text,
        rotor_order: [data.r1_type, data.r2_type, data.r3_type],
        rotor_positions: positions
      });
      return res.data;
    },
  });

  const form = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
    defaultValues: {
      text: '',
      r1_type: 'I',
      r2_type: 'II',
      r3_type: 'III'
    }
  });

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    setPositions([0, 0, 0]);
    mutation.reset();
  };

  const updatePosition = (index: 0 | 1 | 2, val: number) => {
    const newPos: [number, number, number] = [...positions] as [number, number, number];
    newPos[index] = val;
    setPositions(newPos);
  };

  return (
    <ToolPageLayout
      title="Enigma Machine"
      description="Simulate the legendary WWII German Enigma machine. Configure your three rotors and set their initial ring positions (A-Z) to encrypt or decrypt messages."
      icon={RotateCw}
      badges={[{ label: 'Historical', variant: 'warning' }, { label: 'WWII', variant: 'default' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="p-4 rounded-xl border border-border bg-background flex justify-center gap-8">
            <div className="space-y-4">
              <div className="text-center">
                <Label className="text-xs text-muted-foreground uppercase tracking-widest">Fast Rotor (Right)</Label>
                <Select onValueChange={(val) => form.setValue('r3_type', val)} defaultValue={form.getValues('r3_type')}>
                  <SelectTrigger className="bg-card border-border text-foreground w-24 mt-1 mx-auto"><SelectValue /></SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    {['I','II','III','IV','V'].map(r => <SelectItem key={r} value={r} className="text-foreground hover:bg-secondary focus:bg-secondary">{r}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <RotorControl id="r3" label="Rotor III" value={positions[2]} onChange={(v) => updatePosition(2, v)} />
            </div>

            <div className="space-y-4">
              <div className="text-center">
                <Label className="text-xs text-muted-foreground uppercase tracking-widest">Mid Rotor</Label>
                <Select onValueChange={(val) => form.setValue('r2_type', val)} defaultValue={form.getValues('r2_type')}>
                  <SelectTrigger className="bg-card border-border text-foreground w-24 mt-1 mx-auto"><SelectValue /></SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    {['I','II','III','IV','V'].map(r => <SelectItem key={r} value={r} className="text-foreground hover:bg-secondary focus:bg-secondary">{r}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <RotorControl id="r2" label="Rotor II" value={positions[1]} onChange={(v) => updatePosition(1, v)} />
            </div>

            <div className="space-y-4">
              <div className="text-center">
                <Label className="text-xs text-muted-foreground uppercase tracking-widest">Slow Rotor (Left)</Label>
                <Select onValueChange={(val) => form.setValue('r1_type', val)} defaultValue={form.getValues('r1_type')}>
                  <SelectTrigger className="bg-card border-border text-foreground w-24 mt-1 mx-auto"><SelectValue /></SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    {['I','II','III','IV','V'].map(r => <SelectItem key={r} value={r} className="text-foreground hover:bg-secondary focus:bg-secondary">{r}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <RotorControl id="r1" label="Rotor I" value={positions[0]} onChange={(v) => updatePosition(0, v)} />
            </div>
          </div>

          <div className="space-y-3">
            <Label className="text-foreground">Plaintext / Ciphertext</Label>
            <Textarea
              placeholder="Enter text to encrypt or decrypt..."
              className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground font-mono min-h-[120px] uppercase"
              {...form.register('text')}
              onChange={(e) => {
                e.target.value = e.target.value.replace(/[^a-zA-Z\s]/g, '').toUpperCase();
              }}
            />
            {form.formState.errors.text && <p className="text-sm text-destructive">{form.formState.errors.text.message}</p>}
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Execute Machine"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Enigma Output"
        result={res?.result}
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Configure the rotors and input your message to encrypt or decrypt."
      />
    </ToolPageLayout>
  );
}
