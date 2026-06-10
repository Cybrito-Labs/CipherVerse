import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { KeyRound } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import {
  ToolPageLayout,
  ToolInputPanel,
  ToolResultPanel,
  ToolActions
} from '@/components/shared/layout';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { CopyButton } from '@/components/shared/CopyButton';

const schema = z.object({
  payloadStr: z.string().min(2, 'Payload must be a valid JSON object string'),
  secret: z.string().min(1, 'Secret is required'),
  algo: z.enum(['HS256', 'HS384', 'HS512']),
  exp_seconds: z.coerce.number().min(0, 'Must be positive'),
});

type FormValues = z.infer<typeof schema>;

interface JwtResponse {
  token: string;
}

export default function JwtSignPage() {
  const [jsonError, setJsonError] = useState('');

  const mutation = useMutation<JwtResponse, Error, { payload: Record<string, unknown>, secret: string, algo: string, exp_seconds: number }>({
    mutationFn: async (data) => {
      const res = await api.post('/utilities/jwt/sign', data);
      return (res as any).data;
    },
  });

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      payloadStr: '{\n  "sub": "1234567890",\n  "name": "John Doe",\n  "admin": true\n}',
      secret: 'your-256-bit-secret',
      algo: 'HS256',
      exp_seconds: 3600
    }
  });

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    mutation.reset();
    setJsonError('');
  };

  const onSubmit = (d: FormValues) => {
    setJsonError('');
    let parsedPayload: Record<string, unknown>;
    try {
      parsedPayload = JSON.parse(d.payloadStr);
      if (typeof parsedPayload !== 'object' || parsedPayload === null) {
        throw new Error("Payload must be a JSON object");
      }
    } catch (e) {
      setJsonError(`Invalid JSON format: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }

    mutation.mutate({
      payload: parsedPayload,
      secret: d.secret,
      algo: d.algo,
      exp_seconds: d.exp_seconds
    });
  };

  return (
    <ToolPageLayout
      title="JWT Signer"
      description="Cryptographically sign custom JSON payloads to generate JSON Web Tokens (JWT). Perfect for testing API authentication and generating mock tokens."
      icon={KeyRound}
      badges={[{ label: 'Utility', variant: 'default' }, { label: 'Auth', variant: 'success' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <div className="space-y-3">
            <Label className="text-[#EDEDED]">JSON Payload</Label>
            <Textarea
              className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] font-mono min-h-[150px] text-xs"
              {...form.register('payloadStr')}
            />
            {form.formState.errors.payloadStr && <p className="text-sm text-destructive">{form.formState.errors.payloadStr.message}</p>}
            {jsonError && <p className="text-sm text-destructive">{jsonError}</p>}
          </div>

          <div className="space-y-3">
            <Label className="text-[#EDEDED]">Secret Key</Label>
            <Input
              type="password"
              className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] font-mono"
              {...form.register('secret')}
            />
            {form.formState.errors.secret && <p className="text-sm text-destructive">{form.formState.errors.secret.message}</p>}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-3">
              <Label className="text-[#EDEDED]">Algorithm</Label>
              <Select onValueChange={(val) => form.setValue('algo', val as "HS256" | "HS384" | "HS512")} defaultValue={form.getValues('algo')}>
                <SelectTrigger className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED]">
                  <SelectValue placeholder="Algorithm" />
                </SelectTrigger>
                <SelectContent className="bg-[#0A0A0A] border-[#27272A]">
                  <SelectItem value="HS256" className="text-[#EDEDED] hover:bg-[#171717] focus:bg-[#171717]">HMAC SHA-256</SelectItem>
                  <SelectItem value="HS384" className="text-[#EDEDED] hover:bg-[#171717] focus:bg-[#171717]">HMAC SHA-384</SelectItem>
                  <SelectItem value="HS512" className="text-[#EDEDED] hover:bg-[#171717] focus:bg-[#171717]">HMAC SHA-512</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3">
              <Label className="text-[#EDEDED]">Expires In (seconds)</Label>
              <Input
                type="number"
                className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] font-mono"
                {...form.register('exp_seconds')}
              />
            </div>
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit(onSubmit)()}
            onClear={handleClear}
            executeLabel="Sign JWT"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Signed JSON Web Token"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit(onSubmit)()}
        onClear={handleClear}
        emptyMessage="Define your payload and sign it to generate a JWT."
      >
        {res && (
          <div className="space-y-4 pt-2">
            <div className="bg-[#000000] border border-[#27272A] rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">
                  Encoded Token
                </h4>
                <CopyButton text={res.token} />
              </div>

              <div className="font-mono text-sm break-all leading-relaxed bg-[#0A0A0A] p-4 rounded-lg border border-[#27272A]">
                {res.token.split('.').map((part, i) => (
                  <span key={i} className={
                    i === 0 ? 'text-[#F87171] font-bold' :
                    i === 1 ? 'text-primary font-bold' :
                    'text-[#4ADE80] font-bold'
                  }>
                    {part}{i < 2 ? <span className="text-[#52525B]">.</span> : ''}
                  </span>
                ))}
              </div>

              <div className="flex justify-start gap-4 text-xs font-mono mt-4 opacity-70 text-[#A1A1AA]">
                <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-[#F87171]"></div> Header</div>
                <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-primary"></div> Payload</div>
                <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-[#4ADE80]"></div> Signature</div>
              </div>
            </div>
          </div>
        )}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
