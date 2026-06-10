import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { FileSignature, KeyRound, ShieldCheck } from 'lucide-react';
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

const signSchema = z.object({ message: z.string().min(1), private_key: z.string().min(1) });
const verifySchema = z.object({ message: z.string().min(1), signature: z.string().min(1), public_key: z.string().min(1) });

export default function DsaPage() {
  const [activeTab, setActiveTab] = useState('generate');

  const genMutation = useToolExecution<Record<string, never>, RSAKeyResponse>({ endpoint: '/asymmetric/dsa/generate-keys' });
  const signMutation = useToolExecution<z.infer<typeof signSchema>, AsymmetricResponse>({ endpoint: '/asymmetric/dsa/sign' });
  const verifyMutation = useToolExecution<z.infer<typeof verifySchema>, AsymmetricResponse>({ endpoint: '/asymmetric/dsa/verify' });

  const signForm = useForm<z.infer<typeof signSchema>>({ resolver: zodResolver(signSchema) });
  const verifyForm = useForm<z.infer<typeof verifySchema>>({ resolver: zodResolver(verifySchema) });

  const isPending = genMutation.isPending || signMutation.isPending || verifyMutation.isPending;

  const handleClear = () => {
    if (activeTab === 'generate') genMutation.reset();
    if (activeTab === 'sign') { signForm.reset(); signMutation.reset(); }
    if (activeTab === 'verify') { verifyForm.reset(); verifyMutation.reset(); }
  };

  const handleExecute = () => {
    if (activeTab === 'generate') genMutation.mutate({});
    if (activeTab === 'sign') signForm.handleSubmit((d) => signMutation.mutate(d))();
    if (activeTab === 'verify') verifyForm.handleSubmit((d) => verifyMutation.mutate(d))();
  };

  return (
    <ToolPageLayout
      title="DSA Algorithm"
      description="The Digital Signature Algorithm (DSA) is a Federal Information Processing Standard for digital signatures. Unlike RSA, DSA cannot be used for encryption; it is exclusively designed for signing and verification."
      icon={FileSignature}
      badges={[{ label: 'Asymmetric', variant: 'default' }, { label: 'Signature Only', variant: 'warning' }]}
    >
      <ToolInputPanel>
        <ToolTabs
          tabs={[
            { id: 'generate', label: 'Generate Keys' },
            { id: 'sign', label: 'Sign' },
            { id: 'verify', label: 'Verify' },
          ]}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          className="max-w-[400px]"
        />

        {activeTab === 'generate' && (
          <div className="space-y-4">
            <p className="text-sm text-[#A1A1AA] mb-4">Click below to generate a new 2048-bit DSA key pair.</p>
          </div>
        )}

        {activeTab === 'sign' && (
          <form className="space-y-6">
            <div className="space-y-3">
              <Label className="text-[#EDEDED]">Message to Sign</Label>
              <Textarea className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] min-h-[100px]" {...signForm.register('message')} />
            </div>
            <div className="space-y-3">
              <Label className="text-[#EDEDED]">Private Key (PEM format)</Label>
              <Textarea className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] font-mono text-xs h-32" {...signForm.register('private_key')} />
            </div>
          </form>
        )}

        {activeTab === 'verify' && (
          <form className="space-y-6">
            <div className="space-y-3">
              <Label className="text-[#EDEDED]">Original Message</Label>
              <Textarea className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] min-h-[80px]" {...verifyForm.register('message')} />
            </div>
            <div className="space-y-3">
              <Label className="text-[#EDEDED]">Signature (Base64)</Label>
              <Textarea className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] font-mono text-xs min-h-[80px]" {...verifyForm.register('signature')} />
            </div>
            <div className="space-y-3">
              <Label className="text-[#EDEDED]">Public Key (PEM format)</Label>
              <Textarea className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] font-mono text-xs h-32" {...verifyForm.register('public_key')} />
            </div>
          </form>
        )}

        <ToolActions
          isExecuting={isPending}
          onExecute={handleExecute}
          onClear={handleClear}
          executeLabel={
            activeTab === 'generate' ? 'Generate Key Pair' :
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
          emptyMessage="Generate a new DSA key pair to see them here."
        >
          {genMutation.data && (
            <div className="space-y-4 pt-2">
              <div className="bg-[#000000] border border-[#27272A] rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <Label className="text-primary font-semibold">Public Key</Label>
                  <div className="flex gap-2">
                    <CopyButton text={genMutation.data.public_key} />
                    <DownloadButton content={genMutation.data.public_key} filename="dsa_public_key.pem" label="" />
                  </div>
                </div>
                <pre className="text-[11px] text-[#A1A1AA] font-mono p-0 overflow-x-auto h-32">
                  {genMutation.data.public_key}
                </pre>
              </div>
              <div className="bg-[#000000] border border-[#27272A] rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <Label className="text-destructive font-semibold">Private Key (Keep Secret!)</Label>
                  <div className="flex gap-2">
                    <CopyButton text={genMutation.data.private_key} />
                    <DownloadButton content={genMutation.data.private_key} filename="dsa_private_key.pem" label="" />
                  </div>
                </div>
                <pre className="text-[11px] text-[#A1A1AA] font-mono p-0 overflow-x-auto h-32">
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
              <p className="text-sm text-center text-[#A1A1AA]">
                {verifyMutation.data.valid 
                  ? 'The digital signature is valid and mathematically corresponds to the public key and message.' 
                  : 'The digital signature is invalid. The message may have been altered or the wrong key was provided.'}
              </p>
            </div>
          )}
        </ToolResultPanel>
      ) : (
        <ToolResultPanel
          title="Digital Signature"
          result={signMutation.data?.result}
          isLoading={isPending}
          error={signMutation.error}
          onRetry={handleExecute}
          onClear={handleClear}
          emptyMessage="Fill out the input fields and execute to see the results."
        />
      )}
    </ToolPageLayout>
  );
}
