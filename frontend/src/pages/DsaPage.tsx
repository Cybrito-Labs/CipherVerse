import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, FileSignature, KeyRound, ShieldCheck } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { CopyButton } from '@/components/shared/CopyButton';
import { DownloadButton } from '@/components/shared/DownloadButton';
import { useToolExecution } from '@/hooks/useToolExecution';
import type { RSAKeyResponse, AsymmetricResponse } from '@/types/api';

const generateSchema = z.object({});
const signSchema = z.object({ message: z.string().min(1), private_key: z.string().min(1) });
const verifySchema = z.object({ message: z.string().min(1), signature: z.string().min(1), public_key: z.string().min(1) });

export default function DsaPage() {
  const [activeTab, setActiveTab] = useState('generate');

  const genMutation = useToolExecution<{}, RSAKeyResponse>({ endpoint: '/asymmetric/dsa/generate-keys' });
  const signMutation = useToolExecution<z.infer<typeof signSchema>, AsymmetricResponse>({ endpoint: '/asymmetric/dsa/sign' });
  const verifyMutation = useToolExecution<z.infer<typeof verifySchema>, AsymmetricResponse>({ endpoint: '/asymmetric/dsa/verify' });

  const signForm = useForm<z.infer<typeof signSchema>>({ resolver: zodResolver(signSchema) });
  const verifyForm = useForm<z.infer<typeof verifySchema>>({ resolver: zodResolver(verifySchema) });

  const handleClear = () => {
    signForm.reset(); verifyForm.reset();
    genMutation.reset(); signMutation.reset(); verifyMutation.reset();
  };

  const isPending = genMutation.isPending || signMutation.isPending || verifyMutation.isPending;

  return (
    <ToolPageLayout
      title="DSA Algorithm"
      description="The Digital Signature Algorithm (DSA) is a Federal Information Processing Standard for digital signatures. Unlike RSA, DSA cannot be used for encryption; it is exclusively designed for signing and verification."
      icon={FileSignature}
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6">
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="mb-4 bg-background/50 grid w-full grid-cols-3 h-auto py-1">
              <TabsTrigger value="generate" className="py-2">Generate Keys</TabsTrigger>
              <TabsTrigger value="sign" className="py-2">Sign</TabsTrigger>
              <TabsTrigger value="verify" className="py-2">Verify</TabsTrigger>
            </TabsList>

            <TabsContent value="generate" className="space-y-4">
              <p className="text-sm text-muted-foreground mb-4">Click below to generate a new 2048-bit DSA key pair.</p>
              <Button onClick={() => genMutation.mutate({})} disabled={isPending} className="w-full gap-2">
                <KeyRound className="w-4 h-4" /> Generate Key Pair
              </Button>
            </TabsContent>

            <TabsContent value="sign">
              <form onSubmit={signForm.handleSubmit((d) => signMutation.mutate(d))} className="space-y-4">
                <div className="space-y-2">
                  <Label>Message to Sign</Label>
                  <Textarea className="bg-background/50" {...signForm.register('message')} />
                </div>
                <div className="space-y-2">
                  <Label>Private Key (PEM format)</Label>
                  <Textarea className="bg-background/50 font-mono text-xs h-32" {...signForm.register('private_key')} />
                </div>
                <Button type="submit" disabled={isPending} className="w-full gap-2"><Play className="w-4 h-4"/> Generate Signature</Button>
              </form>
            </TabsContent>

            <TabsContent value="verify">
              <form onSubmit={verifyForm.handleSubmit((d) => verifyMutation.mutate(d))} className="space-y-4">
                <div className="space-y-2">
                  <Label>Original Message</Label>
                  <Textarea className="bg-background/50" {...verifyForm.register('message')} />
                </div>
                <div className="space-y-2">
                  <Label>Signature (Base64)</Label>
                  <Textarea className="bg-background/50 font-mono text-xs" {...verifyForm.register('signature')} />
                </div>
                <div className="space-y-2">
                  <Label>Public Key (PEM format)</Label>
                  <Textarea className="bg-background/50 font-mono text-xs h-32" {...verifyForm.register('public_key')} />
                </div>
                <Button type="submit" disabled={isPending} className="w-full gap-2"><ShieldCheck className="w-4 h-4"/> Verify Signature</Button>
              </form>
            </TabsContent>
          </Tabs>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
          {activeTab === 'generate' ? (
            genMutation.data ? (
              <div className="space-y-4">
                <div className="glass rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <Label className="text-primary font-semibold">Public Key</Label>
                    <div className="flex gap-2">
                      <CopyButton text={genMutation.data.public_key} />
                      <DownloadButton content={genMutation.data.public_key} filename="dsa_public_key.pem" label="" />
                    </div>
                  </div>
                  <pre className="text-[10px] font-mono bg-background/50 p-3 rounded border border-border overflow-x-auto h-32">
                    {genMutation.data.public_key}
                  </pre>
                </div>
                <div className="glass rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2">
                    <Label className="text-destructive font-semibold">Private Key (Keep Secret!)</Label>
                    <div className="flex gap-2">
                      <CopyButton text={genMutation.data.private_key} />
                      <DownloadButton content={genMutation.data.private_key} filename="dsa_private_key.pem" label="" />
                    </div>
                  </div>
                  <pre className="text-[10px] font-mono bg-background/50 p-3 rounded border border-border overflow-x-auto h-32">
                    {genMutation.data.private_key}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="glass rounded-xl p-12 text-center text-muted-foreground">
                <KeyRound className="w-12 h-12 mx-auto mb-4 opacity-20" />
                <p>Generated keys will appear here.</p>
              </div>
            )
          ) : activeTab === 'verify' ? (
            <ResultPanel
              title="Verification Result"
              isLoading={verifyMutation.isPending}
              error={verifyMutation.error}
            >
              {verifyMutation.data && (
                <div className={`p-6 rounded-xl border flex flex-col items-center justify-center gap-3 ${verifyMutation.data.valid ? 'bg-success/10 border-success/30' : 'bg-destructive/10 border-destructive/30'}`}>
                  <ShieldCheck className={`w-12 h-12 ${verifyMutation.data.valid ? 'text-success' : 'text-destructive'}`} />
                  <h3 className={`text-xl font-bold ${verifyMutation.data.valid ? 'text-success' : 'text-destructive'}`}>
                    {verifyMutation.data.result}
                  </h3>
                  <p className="text-sm text-center text-muted-foreground">
                    {verifyMutation.data.valid 
                      ? 'The digital signature is valid and mathematically corresponds to the public key and message.' 
                      : 'The digital signature is invalid. The message may have been altered or the wrong key was provided.'}
                  </p>
                </div>
              )}
            </ResultPanel>
          ) : (
            <ResultPanel
              title="Digital Signature"
              result={signMutation.data?.result}
              isLoading={isPending}
              error={signMutation.error}
            />
          )}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
