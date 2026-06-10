import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, Shuffle } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToolExecution } from '@/hooks/useToolExecution';
import type { SymmetricResponse, XORBruteforceResponse } from '@/types/api';

const encryptSchema = z.object({
  text: z.string().min(1, 'Text is required'),
  key: z.string().min(1, 'Key is required'),
});

const decryptSchema = z.object({
  hex_data: z.string().min(1, 'Hex data is required'),
  key: z.string().min(1, 'Key is required'),
});

const bruteSchema = z.object({
  hex_data: z.string().min(1, 'Hex data is required'),
});

export default function XorPage() {
  const [activeTab, setActiveTab] = useState<'encrypt' | 'decrypt' | 'bruteforce'>('encrypt');

  const encryptMutation = useToolExecution<z.infer<typeof encryptSchema>, SymmetricResponse>({ endpoint: '/symmetric/xor/encrypt' });
  const decryptMutation = useToolExecution<z.infer<typeof decryptSchema>, SymmetricResponse>({ endpoint: '/symmetric/xor/decrypt' });
  const bruteMutation = useToolExecution<z.infer<typeof bruteSchema>, XORBruteforceResponse>({ endpoint: '/symmetric/xor/bruteforce' });

  const encryptForm = useForm<z.infer<typeof encryptSchema>>({ resolver: zodResolver(encryptSchema), defaultValues: { text: '', key: '' } });
  const decryptForm = useForm<z.infer<typeof decryptSchema>>({ resolver: zodResolver(decryptSchema), defaultValues: { hex_data: '', key: '' } });
  const bruteForm = useForm<z.infer<typeof bruteSchema>>({ resolver: zodResolver(bruteSchema), defaultValues: { hex_data: '' } });

  const handleClear = () => {
    encryptForm.reset();
    decryptForm.reset();
    bruteForm.reset();
    encryptMutation.reset();
    decryptMutation.reset();
    bruteMutation.reset();
  };

  const isPending = encryptMutation.isPending || decryptMutation.isPending || bruteMutation.isPending;

  return (
    <ToolPageLayout
      title="XOR Cipher"
      description="The XOR cipher applies the exclusive OR bitwise operation to the plaintext and key. It is the basis for many stream ciphers and the One-Time Pad."
      icon={Shuffle}
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6">
          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "encrypt" | "decrypt" | "bruteforce")}>
            <TabsList className="mb-4 bg-background/50 grid w-full grid-cols-3">
              <TabsTrigger value="encrypt">Encrypt</TabsTrigger>
              <TabsTrigger value="decrypt">Decrypt</TabsTrigger>
              <TabsTrigger value="bruteforce">Bruteforce</TabsTrigger>
            </TabsList>

            <TabsContent value="encrypt">
              <form onSubmit={encryptForm.handleSubmit((d) => encryptMutation.mutate(d))} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="enc_text">Text</Label>
                  <Textarea id="enc_text" className="bg-background/50 font-mono" {...encryptForm.register('text')} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="enc_key">Key</Label>
                  <Input id="enc_key" className="bg-background/50" {...encryptForm.register('key')} />
                </div>
                <div className="flex items-center gap-3 pt-2">
                  <Button type="submit" disabled={isPending} className="bg-primary text-primary-foreground gap-2"><Play className="w-4 h-4"/> Encrypt</Button>
                  <Button type="button" variant="outline" onClick={handleClear} className="gap-2"><RotateCcw className="w-4 h-4"/> Clear</Button>
                </div>
              </form>
            </TabsContent>

            <TabsContent value="decrypt">
              <form onSubmit={decryptForm.handleSubmit((d) => decryptMutation.mutate(d))} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="dec_hex">Hex Data</Label>
                  <Textarea id="dec_hex" className="bg-background/50 font-mono" {...decryptForm.register('hex_data')} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="dec_key">Key</Label>
                  <Input id="dec_key" className="bg-background/50" {...decryptForm.register('key')} />
                </div>
                <div className="flex items-center gap-3 pt-2">
                  <Button type="submit" disabled={isPending} className="bg-primary text-primary-foreground gap-2"><Play className="w-4 h-4"/> Decrypt</Button>
                  <Button type="button" variant="outline" onClick={handleClear} className="gap-2"><RotateCcw className="w-4 h-4"/> Clear</Button>
                </div>
              </form>
            </TabsContent>

            <TabsContent value="bruteforce">
              <form onSubmit={bruteForm.handleSubmit((d) => bruteMutation.mutate(d))} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="brute_hex">Hex Data</Label>
                  <Textarea id="brute_hex" className="bg-background/50 font-mono" {...bruteForm.register('hex_data')} />
                  <p className="text-xs text-muted-foreground">Bruteforces a single-byte XOR key.</p>
                </div>
                <div className="flex items-center gap-3 pt-2">
                  <Button type="submit" disabled={isPending} className="bg-primary text-primary-foreground gap-2"><Play className="w-4 h-4"/> Bruteforce</Button>
                  <Button type="button" variant="outline" onClick={handleClear} className="gap-2"><RotateCcw className="w-4 h-4"/> Clear</Button>
                </div>
              </form>
            </TabsContent>
          </Tabs>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
          {activeTab !== 'bruteforce' ? (
            <ResultPanel
              title={activeTab === 'encrypt' ? 'Encrypted Result' : 'Decrypted Result'}
              result={activeTab === 'encrypt' ? encryptMutation.data?.result : decryptMutation.data?.result}
              isLoading={isPending}
              error={activeTab === 'encrypt' ? encryptMutation.error : decryptMutation.error}
            />
          ) : (
            <ResultPanel
              title="Bruteforce Results"
              isLoading={bruteMutation.isPending}
              error={bruteMutation.error}
            >
              {bruteMutation.data && (
                <div className="space-y-2 mt-4 max-h-[400px] overflow-auto pr-2">
                  {bruteMutation.data.results.map(([key, text], idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-background/50 border border-border text-sm flex gap-4">
                      <div className="font-mono text-primary font-bold w-12 flex-shrink-0">
                        0x{key.toString(16).padStart(2, '0').toUpperCase()}
                      </div>
                      <div className="font-mono text-foreground break-all">{text}</div>
                    </div>
                  ))}
                </div>
              )}
            </ResultPanel>
          )}
          {!isPending && !encryptMutation.data && !decryptMutation.data && !bruteMutation.data && (
            <div className="glass rounded-xl p-12 text-center mt-6">
              <p className="text-sm text-muted-foreground">Enter your data and click Execute to see results</p>
            </div>
          )}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
