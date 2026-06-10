import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCw } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
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
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 lg:col-span-6 h-fit">
          <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
            
            <div className="p-4 rounded-xl border border-border bg-black/20 flex justify-center gap-8">
              <div className="space-y-4">
                <div className="text-center">
                  <Label className="text-xs text-muted-foreground uppercase tracking-widest">Fast Rotor (Right)</Label>
                  <Select onValueChange={(val) => form.setValue('r3_type', val)} defaultValue={form.getValues('r3_type')}>
                    <SelectTrigger className="bg-background/50 w-24 mt-1 mx-auto"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {['I','II','III','IV','V'].map(r => <SelectItem key={r} value={r}>{r}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <RotorControl id="r3" label="Rotor III" value={positions[2]} onChange={(v) => updatePosition(2, v)} />
              </div>

              <div className="space-y-4">
                <div className="text-center">
                  <Label className="text-xs text-muted-foreground uppercase tracking-widest">Mid Rotor</Label>
                  <Select onValueChange={(val) => form.setValue('r2_type', val)} defaultValue={form.getValues('r2_type')}>
                    <SelectTrigger className="bg-background/50 w-24 mt-1 mx-auto"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {['I','II','III','IV','V'].map(r => <SelectItem key={r} value={r}>{r}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <RotorControl id="r2" label="Rotor II" value={positions[1]} onChange={(v) => updatePosition(1, v)} />
              </div>

              <div className="space-y-4">
                <div className="text-center">
                  <Label className="text-xs text-muted-foreground uppercase tracking-widest">Slow Rotor (Left)</Label>
                  <Select onValueChange={(val) => form.setValue('r1_type', val)} defaultValue={form.getValues('r1_type')}>
                    <SelectTrigger className="bg-background/50 w-24 mt-1 mx-auto"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {['I','II','III','IV','V'].map(r => <SelectItem key={r} value={r}>{r}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <RotorControl id="r1" label="Rotor I" value={positions[0]} onChange={(v) => updatePosition(0, v)} />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Plaintext / Ciphertext</Label>
              <Textarea 
                placeholder="Enter text to encrypt or decrypt..." 
                className="bg-background/50 font-mono min-h-[120px] uppercase" 
                {...form.register('text')} 
                onChange={(e) => {
                  e.target.value = e.target.value.replace(/[^a-zA-Z\s]/g, '').toUpperCase();
                }}
              />
              {form.formState.errors.text && <p className="text-xs text-destructive">{form.formState.errors.text.message}</p>}
            </div>

            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> Execute Machine
              </Button>
              <Button type="button" variant="outline" onClick={handleClear} className="px-3">
                <RotateCw className="w-4 h-4" />
              </Button>
            </div>
          </form>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="lg:col-span-6">
          <ResultPanel
            title="Enigma Output"
            result={res?.result}
            isLoading={mutation.isPending}
            error={mutation.error}
          />
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
