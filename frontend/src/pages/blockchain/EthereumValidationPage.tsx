import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Wallet, CheckCircle2, XCircle } from 'lucide-react';
import { 
  ToolPageLayout, 
  ToolInputPanel, 
  ToolResultPanel, 
  ToolActions 
} from '@/components/shared/layout';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToolExecution } from '@/hooks/useToolExecution';

const schema = z.object({
  address: z.string().min(1, 'Address is required'),
});

interface AddressResponse {
  valid: boolean;
  address_type?: string;
  network?: string;
}

export default function EthereumValidationPage() {
  const mutation = useToolExecution<z.infer<typeof schema>, AddressResponse>({ endpoint: '/blockchain/ethereum/validate' });
  const form = useForm<z.infer<typeof schema>>({ resolver: zodResolver(schema), defaultValues: { address: '' } });

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  const res = mutation.data;

  return (
    <ToolPageLayout
      title="Ethereum Address Validator"
      description="Check if a given string is a valid Ethereum address. Supports verifying the EIP-55 mixed-case checksum encoding."
      icon={Wallet}
      badges={[{ label: 'Blockchain', variant: 'warning' }, { label: 'Validation', variant: 'default' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="space-y-3">
            <Label className="text-foreground">Ethereum Address</Label>
            <Input 
              placeholder="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045" 
              className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground font-mono" 
              {...form.register('address')} 
            />
            {form.formState.errors.address && <p className="text-sm text-destructive">{form.formState.errors.address.message}</p>}
          </div>
          
          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Validate"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Validation Result"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Enter an address to validate its format and checksum."
      >
        {res && (
          <div className={`p-6 rounded-xl border flex flex-col items-center justify-center gap-3 mt-2 ${res.valid ? 'bg-[#14532D]/10 border-[#14532D]/30' : 'bg-[#7F1D1D]/10 border-[#7F1D1D]/30'}`}>
            {res.valid ? <CheckCircle2 className="w-12 h-12 text-[#4ADE80]" /> : <XCircle className="w-12 h-12 text-[#F87171]" />}
            <h3 className={`text-xl font-bold ${res.valid ? 'text-[#4ADE80]' : 'text-[#F87171]'}`}>
              {res.valid ? 'Valid Address' : 'Invalid Address'}
            </h3>
            
            {res.valid && (
              <div className="w-full mt-4 space-y-2">
                <div className="flex justify-between p-2 rounded bg-background border border-border text-sm">
                  <span className="text-muted-foreground">Type:</span>
                  <span className="font-mono font-bold text-foreground">{res.address_type}</span>
                </div>
              </div>
            )}
          </div>
        )}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
