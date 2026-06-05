import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, Bitcoin, CheckCircle2, XCircle } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToolExecution } from '@/hooks/useToolExecution';
import { ResultPanel } from '@/components/shared/ResultPanel';

const schema = z.object({
  address: z.string().min(1, 'Address is required'),
});

interface AddressResponse {
  valid: boolean;
  address_type?: string;
  network?: string;
}

export default function BitcoinValidationPage() {
  const mutation = useToolExecution<z.infer<typeof schema>, AddressResponse>({ endpoint: '/blockchain/bitcoin/validate' });
  const form = useForm<z.infer<typeof schema>>({ resolver: zodResolver(schema), defaultValues: { address: '' } });

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  const res = mutation.data;

  return (
    <ToolPageLayout
      title="Bitcoin Address Validator"
      description="Check if a given string is a valid Bitcoin address. Supports P2PKH (Legacy), P2SH (Nested SegWit), and Bech32 (Native SegWit) formats."
      icon={Bitcoin}
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 h-fit">
          <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-4">
            <div className="space-y-2">
              <Label>Bitcoin Address</Label>
              <Input 
                placeholder="1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2" 
                className="bg-background/50 font-mono" 
                {...form.register('address')} 
              />
              {form.formState.errors.address && <p className="text-xs text-destructive">{form.formState.errors.address.message}</p>}
            </div>
            
            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> Validate
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

        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
          {!res && !mutation.isPending && !mutation.error ? (
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center">
              <Bitcoin className="w-16 h-16 mb-4 opacity-20" />
              <p>Enter an address to validate its format.</p>
            </div>
          ) : (
            <ResultPanel
              title="Validation Result"
              isLoading={mutation.isPending}
              error={mutation.error}
            >
              {res && (
                <div className={`p-6 rounded-xl border flex flex-col items-center justify-center gap-3 ${res.valid ? 'bg-success/10 border-success/30' : 'bg-destructive/10 border-destructive/30'}`}>
                  {res.valid ? <CheckCircle2 className="w-12 h-12 text-success" /> : <XCircle className="w-12 h-12 text-destructive" />}
                  <h3 className={`text-xl font-bold ${res.valid ? 'text-success' : 'text-destructive'}`}>
                    {res.valid ? 'Valid Address' : 'Invalid Address'}
                  </h3>
                  
                  {res.valid && (
                    <div className="w-full mt-4 space-y-2">
                      <div className="flex justify-between p-2 rounded bg-background/50 text-sm">
                        <span className="text-muted-foreground">Type:</span>
                        <span className="font-mono font-bold text-foreground">{res.address_type}</span>
                      </div>
                      <div className="flex justify-between p-2 rounded bg-background/50 text-sm">
                        <span className="text-muted-foreground">Network:</span>
                        <span className="font-mono font-bold text-foreground">{res.network}</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </ResultPanel>
          )}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
