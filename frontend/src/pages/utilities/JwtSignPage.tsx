import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, KeyRound } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { CopyButton } from '@/components/shared/CopyButton';

const schema = z.object({
  payloadStr: z.string().min(2, 'Payload must be a valid JSON object string'),
  secret: z.string().min(1, 'Secret is required'),
  algo: z.enum(['HS256', 'HS384', 'HS512']),
  exp_seconds: z.coerce.number().min(0, 'Must be positive').default(3600),
});

interface JwtResponse {
  token: string;
}

export default function JwtSignPage() {
  const [jsonError, setJsonError] = useState('');

  const mutation = useMutation<JwtResponse, Error, any>({
    mutationFn: async (data) => {
      const res = await api.post('/utilities/jwt/sign', data);
      return res.data;
    },
  });

  const form = useForm<z.infer<typeof schema>>({ 
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

  const onSubmit = (d: z.infer<typeof schema>) => {
    setJsonError('');
    let parsedPayload = {};
    try {
      parsedPayload = JSON.parse(d.payloadStr);
      if (typeof parsedPayload !== 'object' || parsedPayload === null) {
        throw new Error("Payload must be a JSON object");
      }
    } catch (e: any) {
      setJsonError(`Invalid JSON format: ${e.message}`);
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
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 h-fit">
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            
            <div className="space-y-2">
              <Label>JSON Payload</Label>
              <Textarea 
                className="bg-background/50 font-mono min-h-[150px] text-xs" 
                {...form.register('payloadStr')} 
              />
              {form.formState.errors.payloadStr && <p className="text-xs text-destructive">{form.formState.errors.payloadStr.message}</p>}
              {jsonError && <p className="text-xs text-destructive">{jsonError}</p>}
            </div>

            <div className="space-y-2">
              <Label>Secret Key</Label>
              <Input 
                type="password"
                className="bg-background/50 font-mono" 
                {...form.register('secret')} 
              />
              {form.formState.errors.secret && <p className="text-xs text-destructive">{form.formState.errors.secret.message}</p>}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2 pt-2">
                <Label>Algorithm</Label>
                <Select onValueChange={(val) => form.setValue('algo', val as any)} defaultValue={form.getValues('algo')}>
                  <SelectTrigger className="bg-background/50">
                    <SelectValue placeholder="Algorithm" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="HS256">HMAC SHA-256</SelectItem>
                    <SelectItem value="HS384">HMAC SHA-384</SelectItem>
                    <SelectItem value="HS512">HMAC SHA-512</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2 pt-2">
                <Label>Expires In (seconds)</Label>
                <Input 
                  type="number"
                  className="bg-background/50 font-mono" 
                  {...form.register('exp_seconds')} 
                />
              </div>
            </div>

            <div className="flex items-center gap-3 pt-4">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> Sign JWT
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
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center min-h-[400px]">
              <KeyRound className="w-16 h-16 mb-4 opacity-20" />
              <p>Define your payload and sign it to generate a JWT.</p>
            </div>
          ) : (
            <ResultPanel
              title="Signed JSON Web Token"
              isLoading={mutation.isPending}
              error={mutation.error}
            >
              {res && (
                <div className="space-y-4">
                  <div className="glass rounded-xl p-6 border border-border/50 bg-background/50">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">
                        Encoded Token
                      </h4>
                      <CopyButton text={res.token} />
                    </div>
                    
                    {/* Visualizing the JWT parts: Header.Payload.Signature */}
                    <div className="font-mono text-sm break-all leading-relaxed bg-black/20 p-4 rounded-lg border border-border/50 shadow-inner">
                      {res.token.split('.').map((part, i) => (
                        <span key={i} className={
                          i === 0 ? 'text-destructive font-bold' : 
                          i === 1 ? 'text-primary font-bold' : 
                          'text-success font-bold'
                        }>
                          {part}{i < 2 ? <span className="text-muted-foreground">.</span> : ''}
                        </span>
                      ))}
                    </div>

                    <div className="flex justify-start gap-4 text-xs font-mono mt-4 opacity-70">
                      <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-destructive"></div> Header</div>
                      <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-primary"></div> Payload</div>
                      <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-success"></div> Signature</div>
                    </div>
                  </div>
                </div>
              )}
            </ResultPanel>
          )}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
