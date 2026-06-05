import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, FileBadge } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { useToolExecution } from '@/hooks/useToolExecution';
import { CertificateReport, type ParsedX509 } from '@/components/shared/CertificateReport';

const schema = z.object({
  cert_data: z.string().min(1, 'Certificate data is required'),
});

export default function X509Page() {
  const mutation = useToolExecution<z.infer<typeof schema>, ParsedX509>({ endpoint: '/certificates/x509/parse' });
  const form = useForm<z.infer<typeof schema>>({ resolver: zodResolver(schema), defaultValues: { cert_data: '' } });

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  const cert = mutation.data;

  return (
    <ToolPageLayout
      title="X.509 Certificate Parser"
      description="Decode and inspect PEM or DER encoded X.509 certificates. View detailed subject information, issuer details, validity periods, and cryptographic extensions."
      icon={FileBadge}
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="glass rounded-xl p-6 lg:col-span-4 h-fit">
          <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-4">
            <div className="space-y-2">
              <Label>Certificate Data (PEM format)</Label>
              <Textarea 
                placeholder="-----BEGIN CERTIFICATE-----..." 
                className="bg-background/50 font-mono text-xs h-[300px]" 
                {...form.register('cert_data')} 
              />
              {form.formState.errors.cert_data && <p className="text-xs text-destructive">{form.formState.errors.cert_data.message}</p>}
            </div>
            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> {mutation.isPending ? 'Parsing...' : 'Parse Certificate'}
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
              <FileBadge className="w-16 h-16 mb-4 opacity-20" />
              <p>Paste a certificate and click Parse to view the report.</p>
            </div>
          ) : mutation.isPending ? (
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center h-full min-h-[400px]">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
              <p>Analyzing certificate...</p>
            </div>
          ) : cert ? (
            <CertificateReport cert={cert} />
          ) : null}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
