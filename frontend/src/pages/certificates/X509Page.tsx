import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { FileBadge } from 'lucide-react';
import { 
  ToolPageLayout, 
  ToolInputPanel, 
  ToolResultPanel, 
  ToolActions 
} from '@/components/shared/layout';
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
      badges={[{ label: 'Parser', variant: 'default' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="space-y-3">
            <Label className="text-foreground">Certificate Data (PEM format)</Label>
            <Textarea 
              placeholder="-----BEGIN CERTIFICATE-----..." 
              className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground font-mono text-xs h-[300px]" 
              {...form.register('cert_data')} 
            />
            {form.formState.errors.cert_data && <p className="text-sm text-destructive">{form.formState.errors.cert_data.message}</p>}
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Parse Certificate"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Parsed Certificate"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Paste a certificate and click Parse to view the detailed report."
      >
        {cert && <CertificateReport cert={cert} />}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
