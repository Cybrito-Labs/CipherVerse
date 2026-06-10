import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { ShieldCheck } from 'lucide-react';
import { 
  ToolPageLayout, 
  ToolInputPanel, 
  ToolResultPanel, 
  ToolActions 
} from '@/components/shared/layout';
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
      icon={ShieldCheck}
      badges={[{ label: 'Network', variant: 'warning' }, { label: 'Live', variant: 'success' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="space-y-3">
            <Label className="text-foreground">Hostname / Domain</Label>
            <Input 
              placeholder="example.com" 
              className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground" 
              {...form.register('hostname')} 
            />
            {form.formState.errors.hostname && <p className="text-sm text-destructive">{form.formState.errors.hostname.message}</p>}
          </div>
          
          <div className="space-y-3">
            <Label className="text-foreground">Port</Label>
            <Input 
              type="number"
              placeholder="443" 
              className="bg-background border-border focus:border-muted-foreground text-foreground font-mono" 
              {...form.register('port', { valueAsNumber: true })} 
            />
            {form.formState.errors.port && <p className="text-sm text-destructive">{form.formState.errors.port.message}</p>}
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Analyze TLS"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="TLS Certificate Chain"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Enter a domain to fetch and analyze its live certificate."
      >
        {cert && <CertificateReport cert={cert} />}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
