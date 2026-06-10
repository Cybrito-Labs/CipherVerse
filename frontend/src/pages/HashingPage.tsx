import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Hash, ShieldCheck } from 'lucide-react';
import { ToolPageLayout, ToolInputPanel, ToolResultPanel, ToolTabs, ToolActions } from '@/components/shared/layout';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToolExecution } from '@/hooks/useToolExecution';

const hashSchema = z.object({
  text: z.string().min(1, 'Text to hash is required'),
  algorithm: z.string().min(1, 'Algorithm is required'),
});

const hmacSchema = z.object({
  message: z.string().min(1, 'Message is required'),
  key: z.string().min(1, 'Secret key is required'),
  algorithm: z.string().min(1, 'Algorithm is required'),
});

const pbkdf2Schema = z.object({
  password: z.string().min(1, 'Password is required'),
  iterations: z.coerce.number().min(1, 'Iterations must be at least 1').max(10000000, 'Too many iterations'),
  dklen: z.coerce.number().min(16, 'Key length must be at least 16').max(512, 'Key length too large'),
  hash_name: z.string().min(1, 'Hash algorithm is required'),
});

const scryptSchema = z.object({
  password: z.string().min(1, 'Password is required'),
  n: z.coerce.number().min(2, 'N must be at least 2').max(1048576, 'N is too large'),
  r: z.coerce.number().min(1, 'r must be at least 1').max(256, 'r is too large'),
  p: z.coerce.number().min(1, 'p must be at least 1').max(256, 'p is too large'),
  dklen: z.coerce.number().min(16, 'Key length must be at least 16').max(512, 'Key length too large'),
});

const bcryptSchema = z.object({
  text: z.string().min(1, 'Password to hash is required'),
});

interface HashResponse {
  result: string;
}

interface KdfResponse {
  salt: string;
  derived_key: string;
  iterations?: number;
}

export default function HashingPage() {
  const [activeTab, setActiveTab] = useState<'hash' | 'hmac' | 'pbkdf2' | 'scrypt' | 'bcrypt'>('hash');

  const hashMutation = useToolExecution<z.infer<typeof hashSchema>, HashResponse>({ endpoint: '/hashing/hash' });
  const hmacMutation = useToolExecution<z.infer<typeof hmacSchema>, HashResponse>({ endpoint: '/hashing/hmac' });
  const pbkdf2Mutation = useToolExecution<z.infer<typeof pbkdf2Schema>, KdfResponse>({ endpoint: '/hashing/pbkdf2' });
  const scryptMutation = useToolExecution<z.infer<typeof scryptSchema>, KdfResponse>({ endpoint: '/hashing/scrypt' });
  const bcryptMutation = useToolExecution<z.infer<typeof bcryptSchema> & { algorithm: string }, HashResponse>({ endpoint: '/hashing/bcrypt/hash' });

  const hashForm = useForm<z.infer<typeof hashSchema>>({ resolver: zodResolver(hashSchema) as any, defaultValues: { text: '', algorithm: 'sha256' } });
  const hmacForm = useForm<z.infer<typeof hmacSchema>>({ resolver: zodResolver(hmacSchema) as any, defaultValues: { message: '', key: '', algorithm: 'sha256' } });
  const pbkdf2Form = useForm<z.infer<typeof pbkdf2Schema>>({ resolver: zodResolver(pbkdf2Schema) as any, defaultValues: { password: '', iterations: 100000, dklen: 32, hash_name: 'sha256' } });
  const scryptForm = useForm<z.infer<typeof scryptSchema>>({ resolver: zodResolver(scryptSchema) as any, defaultValues: { password: '', n: 16384, r: 8, p: 1, dklen: 64 } });
  const bcryptForm = useForm<z.infer<typeof bcryptSchema>>({ resolver: zodResolver(bcryptSchema) as any, defaultValues: { text: '' } });

  const handleClear = () => {
    hashForm.reset(); hmacForm.reset(); pbkdf2Form.reset(); scryptForm.reset(); bcryptForm.reset();
    hashMutation.reset(); hmacMutation.reset(); pbkdf2Mutation.reset(); scryptMutation.reset(); bcryptMutation.reset();
  };

  const isPending = hashMutation.isPending || hmacMutation.isPending || pbkdf2Mutation.isPending || scryptMutation.isPending || bcryptMutation.isPending;

  return (
    <ToolPageLayout
      title="Hashing & KDFs"
      description="Cryptographic hashing and key derivation functions for data integrity and secure password storage."
      icon={Hash}
      badges={[
        { label: 'Integrity', variant: 'default' },
        { label: 'KDF', variant: 'success' },
      ]}
    >
      <ToolInputPanel>
        <div className="flex flex-col gap-6">
          <ToolTabs
            tabs={[
              { id: 'hash', label: 'Hash Generator' },
              { id: 'hmac', label: 'HMAC' },
              { id: 'pbkdf2', label: 'PBKDF2' },
              { id: 'scrypt', label: 'Scrypt' },
              { id: 'bcrypt', label: 'Bcrypt' }
            ]}
            activeTab={activeTab}
            onTabChange={(v) => {
              setActiveTab(v as any);
              hashMutation.reset(); hmacMutation.reset(); pbkdf2Mutation.reset(); scryptMutation.reset(); bcryptMutation.reset();
            }}
            className="max-w-[500px]"
          />

          {activeTab === 'hash' && (
            <form onSubmit={hashForm.handleSubmit((d: any) => hashMutation.mutate(d))} className="space-y-6">
              <div className="space-y-3">
                <Label htmlFor="text" className="text-foreground">Text to Hash</Label>
                <Textarea
                  id="text"
                  placeholder="Enter text to hash..."
                  className="min-h-[120px] resize-y bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground"
                  {...hashForm.register('text')}
                />
                {hashForm.formState.errors.text && (
                  <p className="text-sm text-destructive">{hashForm.formState.errors.text.message}</p>
                )}
              </div>

              <div className="space-y-3">
                <Label htmlFor="algorithm" className="text-foreground">Algorithm</Label>
                <Select value={hashForm.watch('algorithm')} onValueChange={(value) => hashForm.setValue('algorithm', value)}>
                  <SelectTrigger className="bg-background border-border focus:border-muted-foreground text-foreground">
                    <SelectValue placeholder="Select algorithm" />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    <SelectItem value="sha256" className="text-foreground hover:bg-secondary focus:bg-secondary">SHA-256</SelectItem>
                    <SelectItem value="sha512" className="text-foreground hover:bg-secondary focus:bg-secondary">SHA-512</SelectItem>
                    <SelectItem value="sha1" className="text-foreground hover:bg-secondary focus:bg-secondary">SHA-1</SelectItem>
                    <SelectItem value="md5" className="text-foreground hover:bg-secondary focus:bg-secondary">MD5</SelectItem>
                    <SelectItem value="blake2b" className="text-foreground hover:bg-secondary focus:bg-secondary">BLAKE2b</SelectItem>
                    <SelectItem value="blake2s" className="text-foreground hover:bg-secondary focus:bg-secondary">BLAKE2s</SelectItem>
                    <SelectItem value="sha3_256" className="text-foreground hover:bg-secondary focus:bg-secondary">SHA3-256</SelectItem>
                  </SelectContent>
                </Select>
                {hashForm.formState.errors.algorithm && (
                  <p className="text-sm text-destructive">{hashForm.formState.errors.algorithm.message}</p>
                )}
              </div>

              <ToolActions
                isExecuting={hashMutation.isPending}
                onExecute={() => hashForm.handleSubmit((d: any) => hashMutation.mutate(d))()}
                onClear={handleClear}
                executeLabel="Generate Hash"
              />
            </form>
          )}

          {activeTab === 'hmac' && (
            <form onSubmit={hmacForm.handleSubmit((d: any) => hmacMutation.mutate(d))} className="space-y-6">
              <div className="space-y-3">
                <Label htmlFor="hmac_message" className="text-foreground">Message</Label>
                <Textarea
                  id="hmac_message"
                  placeholder="Enter message to authenticate..."
                  className="min-h-[120px] resize-y bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground"
                  {...hmacForm.register('message')}
                />
                {hmacForm.formState.errors.message && (
                  <p className="text-sm text-destructive">{hmacForm.formState.errors.message.message}</p>
                )}
              </div>

              <div className="space-y-3">
                <Label htmlFor="hmac_key" className="text-foreground">Secret Key</Label>
                <Input
                  id="hmac_key"
                  type="password"
                  placeholder="Enter secret key..."
                  className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground"
                  {...hmacForm.register('key')}
                />
                {hmacForm.formState.errors.key && (
                  <p className="text-sm text-destructive">{hmacForm.formState.errors.key.message}</p>
                )}
              </div>

              <div className="space-y-3">
                <Label htmlFor="hmac_algorithm" className="text-foreground">Hash Algorithm</Label>
                <Select value={hmacForm.watch('algorithm')} onValueChange={(value) => hmacForm.setValue('algorithm', value)}>
                  <SelectTrigger className="bg-background border-border focus:border-muted-foreground text-foreground">
                    <SelectValue placeholder="Select algorithm" />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    <SelectItem value="sha256" className="text-foreground hover:bg-secondary focus:bg-secondary">HMAC-SHA256</SelectItem>
                    <SelectItem value="sha512" className="text-foreground hover:bg-secondary focus:bg-secondary">HMAC-SHA512</SelectItem>
                    <SelectItem value="sha1" className="text-foreground hover:bg-secondary focus:bg-secondary">HMAC-SHA1</SelectItem>
                    <SelectItem value="md5" className="text-foreground hover:bg-secondary focus:bg-secondary">HMAC-MD5</SelectItem>
                  </SelectContent>
                </Select>
                {hmacForm.formState.errors.algorithm && (
                  <p className="text-sm text-destructive">{hmacForm.formState.errors.algorithm.message}</p>
                )}
              </div>

              <ToolActions
                isExecuting={hmacMutation.isPending}
                onExecute={() => hmacForm.handleSubmit((d: any) => hmacMutation.mutate(d))()}
                onClear={handleClear}
                executeLabel="Generate HMAC"
              />
            </form>
          )}

          {activeTab === 'pbkdf2' && (
            <form onSubmit={pbkdf2Form.handleSubmit((d: any) => pbkdf2Mutation.mutate(d))} className="space-y-6">
              <div className="space-y-3">
                <Label htmlFor="pbkdf2_password" className="text-foreground">Password / Passphrase</Label>
                <Input
                  id="pbkdf2_password"
                  type="password"
                  placeholder="Enter password to derive key from..."
                  className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground"
                  {...pbkdf2Form.register('password')}
                />
                {pbkdf2Form.formState.errors.password && (
                  <p className="text-sm text-destructive">{pbkdf2Form.formState.errors.password.message}</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-3">
                  <Label htmlFor="iterations" className="text-foreground">Iterations</Label>
                  <Input
                    id="iterations"
                    type="number"
                    min="1"
                    className="bg-background border-border focus:border-muted-foreground text-foreground font-mono"
                    {...pbkdf2Form.register('iterations')}
                  />
                  {pbkdf2Form.formState.errors.iterations && (
                    <p className="text-sm text-destructive">{pbkdf2Form.formState.errors.iterations.message}</p>
                  )}
                </div>

                <div className="space-y-3">
                  <Label htmlFor="dklen" className="text-foreground">Derived Key Length (bytes)</Label>
                  <Input
                    id="dklen"
                    type="number"
                    min="16"
                    className="bg-background border-border focus:border-muted-foreground text-foreground font-mono"
                    {...pbkdf2Form.register('dklen')}
                  />
                  {pbkdf2Form.formState.errors.dklen && (
                    <p className="text-sm text-destructive">{pbkdf2Form.formState.errors.dklen.message}</p>
                  )}
                </div>
              </div>

              <div className="space-y-3">
                <Label htmlFor="hash_name" className="text-foreground">Underlying Hash</Label>
                <Select value={pbkdf2Form.watch('hash_name')} onValueChange={(value) => pbkdf2Form.setValue('hash_name', value)}>
                  <SelectTrigger className="bg-background border-border focus:border-muted-foreground text-foreground">
                    <SelectValue placeholder="Select hash" />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    <SelectItem value="sha256" className="text-foreground hover:bg-secondary focus:bg-secondary">SHA-256</SelectItem>
                    <SelectItem value="sha512" className="text-foreground hover:bg-secondary focus:bg-secondary">SHA-512</SelectItem>
                    <SelectItem value="sha1" className="text-foreground hover:bg-secondary focus:bg-secondary">SHA-1</SelectItem>
                  </SelectContent>
                </Select>
                {pbkdf2Form.formState.errors.hash_name && (
                  <p className="text-sm text-destructive">{pbkdf2Form.formState.errors.hash_name.message}</p>
                )}
              </div>

              <ToolActions
                isExecuting={pbkdf2Mutation.isPending}
                onExecute={() => pbkdf2Form.handleSubmit((d: any) => pbkdf2Mutation.mutate(d))()}
                onClear={handleClear}
                executeLabel="Derive Key"
              />
            </form>
          )}

          {activeTab === 'scrypt' && (
            <form onSubmit={scryptForm.handleSubmit((d: any) => scryptMutation.mutate(d))} className="space-y-6">
              <div className="space-y-3">
                <Label htmlFor="scrypt_password" className="text-foreground">Password / Passphrase</Label>
                <Input
                  id="scrypt_password"
                  type="password"
                  placeholder="Enter password to derive key from..."
                  className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground"
                  {...scryptForm.register('password')}
                />
                {scryptForm.formState.errors.password && (
                  <p className="text-sm text-destructive">{scryptForm.formState.errors.password.message}</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-3">
                  <Label htmlFor="n" className="text-foreground">N (Cost Parameter)</Label>
                  <Input
                    id="n"
                    type="number"
                    min="2"
                    className="bg-background border-border focus:border-muted-foreground text-foreground font-mono"
                    {...scryptForm.register('n')}
                  />
                  <p className="text-[11px] text-muted-foreground">CPU/Memory cost (power of 2)</p>
                  {scryptForm.formState.errors.n && (
                    <p className="text-sm text-destructive">{scryptForm.formState.errors.n.message}</p>
                  )}
                </div>

                <div className="space-y-3">
                  <Label htmlFor="r" className="text-foreground">r (Block Size)</Label>
                  <Input
                    id="r"
                    type="number"
                    min="1"
                    className="bg-background border-border focus:border-muted-foreground text-foreground font-mono"
                    {...scryptForm.register('r')}
                  />
                  <p className="text-[11px] text-muted-foreground">Block size parameter</p>
                  {scryptForm.formState.errors.r && (
                    <p className="text-sm text-destructive">{scryptForm.formState.errors.r.message}</p>
                  )}
                </div>

                <div className="space-y-3">
                  <Label htmlFor="p" className="text-foreground">p (Parallelization)</Label>
                  <Input
                    id="p"
                    type="number"
                    min="1"
                    className="bg-background border-border focus:border-muted-foreground text-foreground font-mono"
                    {...scryptForm.register('p')}
                  />
                  {scryptForm.formState.errors.p && (
                    <p className="text-sm text-destructive">{scryptForm.formState.errors.p.message}</p>
                  )}
                </div>

                <div className="space-y-3">
                  <Label htmlFor="scrypt_dklen" className="text-foreground">Derived Key Length</Label>
                  <Input
                    id="scrypt_dklen"
                    type="number"
                    min="16"
                    className="bg-background border-border focus:border-muted-foreground text-foreground font-mono"
                    {...scryptForm.register('dklen')}
                  />
                  {scryptForm.formState.errors.dklen && (
                    <p className="text-sm text-destructive">{scryptForm.formState.errors.dklen.message}</p>
                  )}
                </div>
              </div>

              <ToolActions
                isExecuting={scryptMutation.isPending}
                onExecute={() => scryptForm.handleSubmit((d: any) => scryptMutation.mutate(d))()}
                onClear={handleClear}
                executeLabel="Derive Key"
              />
            </form>
          )}

          {activeTab === 'bcrypt' && (
            <form onSubmit={bcryptForm.handleSubmit((d: any) => bcryptMutation.mutate({ ...d, algorithm: 'bcrypt' }))} className="space-y-6">
              <div className="space-y-3">
                <Label htmlFor="bcrypt_text" className="text-foreground">Password</Label>
                <Input
                  id="bcrypt_text"
                  type="password"
                  placeholder="Enter password to hash..."
                  className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground"
                  {...bcryptForm.register('text')}
                />
                <p className="text-[11px] text-muted-foreground">Bcrypt automatically handles salt generation and multiple rounds.</p>
                {bcryptForm.formState.errors.text && (
                  <p className="text-sm text-destructive">{bcryptForm.formState.errors.text.message}</p>
                )}
              </div>

              <ToolActions
                isExecuting={bcryptMutation.isPending}
                onExecute={() => bcryptForm.handleSubmit((d: any) => bcryptMutation.mutate({ ...d, algorithm: 'bcrypt' }))()}
                onClear={handleClear}
                executeLabel="Generate Hash"
              />
            </form>
          )}
        </div>
      </ToolInputPanel>

      <ToolResultPanel
        title={
          activeTab === 'hash' ? 'Hash Digest' : 
          activeTab === 'hmac' ? 'HMAC Digest' : 
          activeTab === 'pbkdf2' || activeTab === 'scrypt' ? 'Derived Key (Hex)' : 
          'Bcrypt Hash'
        }
        result={
          activeTab === 'hash' ? hashMutation.data?.result :
          activeTab === 'hmac' ? hmacMutation.data?.result :
          activeTab === 'pbkdf2' ? pbkdf2Mutation.data?.derived_key :
          activeTab === 'scrypt' ? scryptMutation.data?.derived_key :
          bcryptMutation.data?.result
        }
        isLoading={isPending}
        error={
          activeTab === 'hash' ? hashMutation.error :
          activeTab === 'hmac' ? hmacMutation.error :
          activeTab === 'pbkdf2' ? pbkdf2Mutation.error :
          activeTab === 'scrypt' ? scryptMutation.error :
          bcryptMutation.error
        }
        onRetry={
          activeTab === 'hash' ? () => hashForm.handleSubmit((d: any) => hashMutation.mutate(d))() :
          activeTab === 'hmac' ? () => hmacForm.handleSubmit((d: any) => hmacMutation.mutate(d))() :
          activeTab === 'pbkdf2' ? () => pbkdf2Form.handleSubmit((d: any) => pbkdf2Mutation.mutate(d))() :
          activeTab === 'scrypt' ? () => scryptForm.handleSubmit((d: any) => scryptMutation.mutate(d))() :
          () => bcryptForm.handleSubmit((d: any) => bcryptMutation.mutate({ ...d, algorithm: 'bcrypt' }))()
        }
        onClear={handleClear}
        emptyMessage={
          activeTab === 'hash' ? "Enter your text and select an algorithm to generate a hash fingerprint" :
          activeTab === 'hmac' ? "Provide a message and secret key to compute the HMAC" :
          activeTab === 'pbkdf2' ? "Configure PBKDF2 parameters and derive a secure cryptographic key from a password" :
          activeTab === 'scrypt' ? "Configure Scrypt parameters and derive a memory-hard cryptographic key" :
          "Enter a password to generate a secure, salted Bcrypt hash"
        }
      >
        {activeTab === 'pbkdf2' && pbkdf2Mutation.data && (
          <div className="mt-4 border-t border-border pt-4 space-y-3">
            <div className="flex items-center gap-2 mb-2">
              <ShieldCheck className="w-4 h-4 text-[#14532D]" />
              <h4 className="text-[13px] font-semibold text-foreground">Security Report</h4>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-background border border-border rounded-lg p-3">
                <div className="text-[11px] text-muted-foreground uppercase tracking-wider mb-1">Generated Salt</div>
                <div className="text-[13px] font-mono text-foreground truncate">{pbkdf2Mutation.data.salt}</div>
              </div>
              <div className="bg-background border border-border rounded-lg p-3">
                <div className="text-[11px] text-muted-foreground uppercase tracking-wider mb-1">Iterations</div>
                <div className="text-[13px] font-mono text-foreground">{pbkdf2Form.getValues('iterations').toLocaleString()}</div>
              </div>
            </div>
          </div>
        )}
        {activeTab === 'scrypt' && scryptMutation.data && (
          <div className="mt-4 border-t border-border pt-4 space-y-3">
            <div className="flex items-center gap-2 mb-2">
              <ShieldCheck className="w-4 h-4 text-[#14532D]" />
              <h4 className="text-[13px] font-semibold text-foreground">Security Report</h4>
            </div>
            <div className="bg-background border border-border rounded-lg p-3">
              <div className="text-[11px] text-muted-foreground uppercase tracking-wider mb-1">Generated Salt</div>
              <div className="text-[13px] font-mono text-foreground truncate">{scryptMutation.data.salt}</div>
            </div>
          </div>
        )}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
