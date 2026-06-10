import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Lock, KeyRound, ShieldCheck } from 'lucide-react';
import { 
  ToolPageLayout, 
  ToolInputPanel, 
  ToolResultPanel, 
  ToolTabs, 
  ToolActions 
} from '@/components/shared/layout';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { CopyButton } from '@/components/shared/CopyButton';
import { DownloadButton } from '@/components/shared/DownloadButton';
import { useToolExecution } from '@/hooks/useToolExecution';
import type { RSAKeyResponse, AsymmetricResponse } from '@/types/api';

const encryptSchema = z.object({ text: z.string().min(1), public_key: z.string().min(1) });
const decryptSchema = z.object({ encrypted_base64: z.string().min(1), private_key: z.string().min(1) });
const signSchema = z.object({ message: z.string().min(1), private_key: z.string().min(1) });
const verifySchema = z.object({ message: z.string().min(1), signature: z.string().min(1), public_key: z.string().min(1) });

export default function RsaPage() {
  const [activeTab, setActiveTab] = useState('generate');

  const genMutation = useToolExecution<Record<string, never>, RSAKeyResponse>({ endpoint: '/asymmetric/rsa/generate-keys' });
  const encMutation = useToolExecution<z.infer<typeof encryptSchema>, AsymmetricResponse>({ endpoint: '/asymmetric/rsa/encrypt' });
  const decMutation = useToolExecution<z.infer<typeof decryptSchema>, AsymmetricResponse>({ endpoint: '/asymmetric/rsa/decrypt' });
  const signMutation = useToolExecution<z.infer<typeof signSchema>, AsymmetricResponse>({ endpoint: '/asymmetric/rsa/sign' });
  const verifyMutation = useToolExecution<z.infer<typeof verifySchema>, AsymmetricResponse>({ endpoint: '/asymmetric/rsa/verify' });

  const encForm = useForm<z.infer<typeof encryptSchema>>({ resolver: zodResolver(encryptSchema) });
  const decForm = useForm<z.infer<typeof decryptSchema>>({ resolver: zodResolver(decryptSchema) });
  const signForm = useForm<z.infer<typeof signSchema>>({ resolver: zodResolver(signSchema) });
  const verifyForm = useForm<z.infer<typeof verifySchema>>({ resolver: zodResolver(verifySchema) });

  const isPending = genMutation.isPending || encMutation.isPending || decMutation.isPending || signMutation.isPending || verifyMutation.isPending;

  const handleClear = () => {
    if (activeTab === 'generate') genMutation.reset();
    if (activeTab === 'encrypt') { encForm.reset(); encMutation.reset(); }
    if (activeTab === 'decrypt') { decForm.reset(); decMutation.reset(); }
    if (activeTab === 'sign') { signForm.reset(); signMutation.reset(); }
    if (activeTab === 'verify') { verifyForm.reset(); verifyMutation.reset(); }
  };

  const handleExecute = () => {
    if (activeTab === 'generate') genMutation.mutate({});
    if (activeTab === 'encrypt') encForm.handleSubmit((d) => encMutation.mutate(d))();
    if (activeTab === 'decrypt') decForm.handleSubmit((d) => decMutation.mutate(d))();
    if (activeTab === 'sign') signForm.handleSubmit((d) => signMutation.mutate(d))();
    if (activeTab === 'verify') verifyForm.handleSubmit((d) => verifyMutation.mutate(d))();
  };

  return (
    <ToolPageLayout
      title="RSA Algorithm"
      description="Rivest-Shamir-Adleman (RSA) is a widely used public-key cryptosystem. Use this tool to generate key pairs, encrypt data with a public key, decrypt with a private key, and create or verify digital signatures."
      icon={Lock}
      badges={[{ label: 'Asymmetric', variant: 'default' }]}
    >
      <ToolInputPanel>
        <ToolTabs
          tabs={[
            { id: 'generate', label: 'Generate' },
            { id: 'encrypt', label: 'Encrypt' },
            { id: 'decrypt', label: 'Decrypt' },
            { id: 'sign', label: 'Sign' },
            { id: 'verify', label: 'Verify' },
          ]}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          className="max-w-full"
        />

        {activeTab === 'generate' && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground mb-4">Click below to generate a new 2048-bit RSA key pair.</p>
          </div>
        )}

        {activeTab === 'encrypt' && (
          <form className="space-y-6">
            <div className="space-y-3">
              <Label className="text-foreground">Plaintext</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground min-h-[100px]" {...encForm.register('text')} />
            </div>
            <div className="space-y-3">
              <Label className="text-foreground">Public Key (PEM format)</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground font-mono text-xs h-32" {...encForm.register('public_key')} />
            </div>
          </form>
        )}

        {activeTab === 'decrypt' && (
          <form className="space-y-6">
            <div className="space-y-3">
              <Label className="text-foreground">Ciphertext (Base64)</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground font-mono text-xs min-h-[100px]" {...decForm.register('encrypted_base64')} />
            </div>
            <div className="space-y-3">
              <Label className="text-foreground">Private Key (PEM format)</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground font-mono text-xs h-32" {...decForm.register('private_key')} />
            </div>
          </form>
        )}

        {activeTab === 'sign' && (
          <form className="space-y-6">
            <div className="space-y-3">
              <Label className="text-foreground">Message to Sign</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground min-h-[100px]" {...signForm.register('message')} />
            </div>
            <div className="space-y-3">
              <Label className="text-foreground">Private Key (PEM format)</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground font-mono text-xs h-32" {...signForm.register('private_key')} />
            </div>
          </form>
        )}

        {activeTab === 'verify' && (
          <form className="space-y-6">
            <div className="space-y-3">
              <Label className="text-foreground">Original Message</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground min-h-[80px]" {...verifyForm.register('message')} />
            </div>
            <div className="space-y-3">
              <Label className="text-foreground">Signature (Base64)</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground font-mono text-xs min-h-[80px]" {...verifyForm.register('signature')} />
            </div>
            <div className="space-y-3">
              <Label className="text-foreground">Public Key (PEM format)</Label>
              <Textarea className="bg-background border-border focus:border-muted-foreground text-foreground font-mono text-xs h-32" {...verifyForm.register('public_key')} />
            </div>
          </form>
        )}

        <ToolActions
          isExecuting={isPending}
          onExecute={handleExecute}
          onClear={handleClear}
          executeLabel={
            activeTab === 'generate' ? 'Generate Key Pair' :
            activeTab === 'encrypt' ? 'Encrypt' :
            activeTab === 'decrypt' ? 'Decrypt' :
            activeTab === 'sign' ? 'Generate Signature' : 'Verify Signature'
          }
          executeIcon={activeTab === 'generate' ? KeyRound : activeTab === 'verify' ? ShieldCheck : undefined}
        />
      </ToolInputPanel>

      {activeTab === 'generate' ? (
        <ToolResultPanel
          title="Generated Keys"
          isLoading={genMutation.isPending}
          error={genMutation.error}
          onRetry={handleExecute}
          onClear={handleClear}
          emptyMessage="Generate a new RSA key pair to see them here."
        >
          {genMutation.data && (
            <div className="space-y-4 pt-2">
              <div className="bg-background border border-border rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <Label className="text-primary font-semibold">Public Key</Label>
                  <div className="flex gap-2">
                    <CopyButton text={genMutation.data.public_key} />
                    <DownloadButton content={genMutation.data.public_key} filename="public_key.pem" label="" />
                  </div>
                </div>
                <pre className="text-[11px] text-muted-foreground font-mono p-0 overflow-x-auto h-32">
                  {genMutation.data.public_key}
                </pre>
              </div>
              <div className="bg-background border border-border rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <Label className="text-destructive font-semibold">Private Key (Keep Secret!)</Label>
                  <div className="flex gap-2">
                    <CopyButton text={genMutation.data.private_key} />
                    <DownloadButton content={genMutation.data.private_key} filename="private_key.pem" label="" />
                  </div>
                </div>
                <pre className="text-[11px] text-muted-foreground font-mono p-0 overflow-x-auto h-32">
                  {genMutation.data.private_key}
                </pre>
              </div>
            </div>
          )}
        </ToolResultPanel>
      ) : activeTab === 'verify' ? (
        <ToolResultPanel
          title="Verification Result"
          isLoading={verifyMutation.isPending}
          error={verifyMutation.error}
          onRetry={handleExecute}
          onClear={handleClear}
          emptyMessage="Provide the original message, signature, and public key to verify authenticity."
        >
          {verifyMutation.data && (
            <div className={`p-6 rounded-xl border flex flex-col items-center justify-center gap-3 mt-2 ${verifyMutation.data.valid ? 'bg-[#14532D]/10 border-[#14532D]/30' : 'bg-[#7F1D1D]/10 border-[#7F1D1D]/30'}`}>
              <ShieldCheck className={`w-12 h-12 ${verifyMutation.data.valid ? 'text-[#4ADE80]' : 'text-[#F87171]'}`} />
              <h3 className={`text-xl font-bold ${verifyMutation.data.valid ? 'text-[#4ADE80]' : 'text-[#F87171]'}`}>
                {verifyMutation.data.result}
              </h3>
              <p className="text-sm text-center text-muted-foreground">
                {verifyMutation.data.valid 
                  ? 'The digital signature mathematically matches the public key and message.' 
                  : 'The digital signature is invalid. The message may have been tampered with or the wrong key was used.'}
              </p>
            </div>
          )}
        </ToolResultPanel>
      ) : (
        <ToolResultPanel
          title={activeTab === 'encrypt' ? 'Encrypted Ciphertext' : activeTab === 'decrypt' ? 'Decrypted Plaintext' : 'Digital Signature'}
          result={
            activeTab === 'encrypt' ? encMutation.data?.result :
            activeTab === 'decrypt' ? decMutation.data?.result :
            signMutation.data?.result
          }
          isLoading={isPending}
          error={
            activeTab === 'encrypt' ? encMutation.error :
            activeTab === 'decrypt' ? decMutation.error :
            signMutation.error
          }
          onRetry={handleExecute}
          onClear={handleClear}
          emptyMessage="Fill out the input fields and execute to see the results."
        />
      )}
    </ToolPageLayout>
  );
}
