import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCw, Calculator } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
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
  // Typex uses 5 rotors instead of Enigma's 3
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
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 lg:col-span-8 h-fit">
          <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
            
            <div className="p-4 rounded-xl border border-border bg-black/20 overflow-x-auto">
              <div className="flex justify-center gap-4 min-w-max pb-2">
                {/* Rendering 5 rotors right to left (Rotor 5 is the fastest) */}
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

            <div className="space-y-2">
              <Label>Plaintext / Ciphertext</Label>
              <Textarea 
                placeholder="Enter text to encrypt or decrypt..." 
                className="bg-background/50 font-mono min-h-[120px] uppercase" 
                {...form.register('text')} 
                onChange={(e) => e.target.value = e.target.value.replace(/[^a-zA-Z\s]/g, '').toUpperCase()}
              />
              {form.formState.errors.text && <p className="text-xs text-destructive">{form.formState.errors.text.message}</p>}
            </div>

            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> Execute Typex
              </Button>
              <Button type="button" variant="outline" onClick={handleClear} className="px-3">
                <RotateCw className="w-4 h-4" />
              </Button>
            </div>
          </form>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="lg:col-span-4">
          <ResultPanel
            title="Typex Output"
            result={res?.result}
            isLoading={mutation.isPending}
            error={mutation.error}
          />
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
