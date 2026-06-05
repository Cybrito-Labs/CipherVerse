import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, ShieldSearch } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToolExecution } from '@/hooks/useToolExecution';
import { CertificateReport, type ParsedX509 } from '@/components/shared/CertificateReport';

const schema = z.object({
  hostname: z.string().min(1, 'Hostname is required'),
  port: z.number().int().min(1).max(65535),
});

export default function TlsPage() {
  const mutation = useToolExecution<z.infer<typeof schema>, ParsedX509>({ endpoint: '/certificates/tls/analyze' });
  const form = useForm<z.infer<typeof schema>>({ 
    resolver: zodResolver(schema), 
    defaultValues: { hostname: '', port: 443 } 
  });

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  const cert = mutation.data;

  return (
    <ToolPageLayout
      title="TLS Analyzer"
      description="Connect to a remote server and analyze its SSL/TLS certificate chain. Extracts the X.509 certificate directly from the live connection to verify its current status, validity, and cryptographic properties."
      icon={ShieldSearch}
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-xl p-6 lg:col-span-4 h-fit">
          <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-4">
            <div className="space-y-2">
              <Label>Hostname / Domain</Label>
              <Input 
                placeholder="example.com" 
                className="bg-background/50" 
                {...form.register('hostname')} 
              />
              {form.formState.errors.hostname && <p className="text-xs text-destructive">{form.formState.errors.hostname.message}</p>}
            </div>
            
            <div className="space-y-2">
              <Label>Port</Label>
              <Input 
                type="number"
                placeholder="443" 
                className="bg-background/50 font-mono" 
                {...form.register('port', { valueAsNumber: true })} 
              />
              {form.formState.errors.port && <p className="text-xs text-destructive">{form.formState.errors.port.message}</p>}
            </div>

            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> {mutation.isPending ? 'Connecting...' : 'Analyze TLS'}
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

        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="lg:col-span-8">
          {!cert && !mutation.isPending ? (
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center h-full min-h-[400px]">
              <ShieldSearch className="w-16 h-16 mb-4 opacity-20" />
              <p>Enter a domain to fetch and analyze its live certificate.</p>
            </div>
          ) : mutation.isPending ? (
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center h-full min-h-[400px]">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
              <p>Establishing secure connection and retrieving certificate...</p>
            </div>
          ) : cert ? (
            <CertificateReport cert={cert} />
          ) : null}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
