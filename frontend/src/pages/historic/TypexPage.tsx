import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Calculator } from 'lucide-react';
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
import { RotorControl } from '@/components/shared/RotorControl';

const schema = z.object({
  text: z.string().min(1, 'Text is required').regex(/^[a-zA-Z\s]+$/, "Only letters and spaces allowed"),
});

interface TypexResponse {
  result: string;
}

export default function TypexPage() {
  const [positions, setPositions] = useState<number[]>([0, 0, 0, 0, 0]);

  const mutation = useMutation<TypexResponse, Error, z.infer<typeof schema>>({
    mutationFn: async (data) => {
      const res = await api.post('/historic/typex', {
        text: data.text,
        rotors: 5,
        positions: positions
      });
      return res.data;
    },
  });

  const form = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
    defaultValues: { text: '' }
  });

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    setPositions([0, 0, 0, 0, 0]);
    mutation.reset();
  };

  const updatePosition = (index: number, val: number) => {
    const newPos = [...positions];
    newPos[index] = val;
    setPositions(newPos);
  };

  return (
    <ToolPageLayout
      title="Typex Machine Simulator"
      description="The British Typex was a heavily modified Enigma variant with enhanced security. It utilized 5 rotors instead of 3, rendering it virtually unbreakable during WWII."
      icon={Calculator}
      badges={[{ label: 'Historical', variant: 'warning' }, { label: '5-Rotor', variant: 'default' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="p-4 rounded-xl border border-border bg-background overflow-x-auto">
            <div className="flex justify-center gap-4 min-w-max pb-2">
              {[4, 3, 2, 1, 0].map((rotorIdx) => (
                <RotorControl
                  key={rotorIdx}
                  id={`r${rotorIdx}`}
                  label={`Rotor ${rotorIdx + 1}`}
                  value={positions[rotorIdx]}
                  onChange={(v) => updatePosition(rotorIdx, v)}
                />
              ))}
            </div>
          </div>

          <div className="space-y-3">
            <Label className="text-foreground">Plaintext / Ciphertext</Label>
            <Textarea
              placeholder="Enter text to encrypt or decrypt..."
              className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground font-mono min-h-[120px] uppercase"
              {...form.register('text')}
              onChange={(e) => e.target.value = e.target.value.replace(/[^a-zA-Z\s]/g, '').toUpperCase()}
            />
            {form.formState.errors.text && <p className="text-sm text-destructive">{form.formState.errors.text.message}</p>}
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Execute Typex"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Typex Output"
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
