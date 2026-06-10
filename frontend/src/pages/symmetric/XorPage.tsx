import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Shuffle } from 'lucide-react';
import { 
  ToolPageLayout, 
  ToolInputPanel, 
  ToolResultPanel, 
  ToolTabs, 
  ToolActions 
} from '@/components/shared/layout';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
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
      <ToolInputPanel>
        <div className="flex flex-col gap-6">
          <ToolTabs
            tabs={[
              { id: 'encrypt', label: 'Encrypt' },
              { id: 'decrypt', label: 'Decrypt' },
              { id: 'bruteforce', label: 'Bruteforce' }
            ]}
            activeTab={activeTab}
            onTabChange={(v) => {
              setActiveTab(v as 'encrypt' | 'decrypt' | 'bruteforce');
              encryptMutation.reset();
              decryptMutation.reset();
              bruteMutation.reset();
            }}
            className="max-w-[350px]"
          />

          {activeTab === 'encrypt' && (
            <form onSubmit={encryptForm.handleSubmit((d) => encryptMutation.mutate(d))} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="enc_text" className="text-[#EDEDED]">Text</Label>
                <Textarea 
                  id="enc_text" 
                  placeholder="Enter text to encrypt..."
                  className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] placeholder:text-[#52525B] min-h-[100px] font-mono text-sm" 
                  {...encryptForm.register('text')} 
                />
                {encryptForm.formState.errors.text && (
                  <p className="text-sm text-destructive">{encryptForm.formState.errors.text.message}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="enc_key" className="text-[#EDEDED]">Key</Label>
                <Input 
                  id="enc_key" 
                  placeholder="Enter encryption key..."
                  className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] placeholder:text-[#52525B] font-mono" 
                  {...encryptForm.register('key')} 
                />
                {encryptForm.formState.errors.key && (
                  <p className="text-sm text-destructive">{encryptForm.formState.errors.key.message}</p>
                )}
              </div>
              <ToolActions
                isExecuting={encryptMutation.isPending}
                onExecute={() => encryptForm.handleSubmit((d) => encryptMutation.mutate(d))()}
                onClear={handleClear}
                executeLabel="Encrypt"
              />
            </form>
          )}

          {activeTab === 'decrypt' && (
            <form onSubmit={decryptForm.handleSubmit((d) => decryptMutation.mutate(d))} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="dec_hex" className="text-[#EDEDED]">Hex Data</Label>
                <Textarea 
                  id="dec_hex" 
                  placeholder="Enter Hex data to decrypt..."
                  className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] placeholder:text-[#52525B] min-h-[100px] font-mono text-sm" 
                  {...decryptForm.register('hex_data')} 
                />
                {decryptForm.formState.errors.hex_data && (
                  <p className="text-sm text-destructive">{decryptForm.formState.errors.hex_data.message}</p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="dec_key" className="text-[#EDEDED]">Key</Label>
                <Input 
                  id="dec_key" 
                  placeholder="Enter decryption key..."
                  className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] placeholder:text-[#52525B] font-mono" 
                  {...decryptForm.register('key')} 
                />
                {decryptForm.formState.errors.key && (
                  <p className="text-sm text-destructive">{decryptForm.formState.errors.key.message}</p>
                )}
              </div>
              <ToolActions
                isExecuting={decryptMutation.isPending}
                onExecute={() => decryptForm.handleSubmit((d) => decryptMutation.mutate(d))()}
                onClear={handleClear}
                executeLabel="Decrypt"
              />
            </form>
          )}

          {activeTab === 'bruteforce' && (
            <form onSubmit={bruteForm.handleSubmit((d) => bruteMutation.mutate(d))} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="brute_hex" className="text-[#EDEDED]">Hex Data</Label>
                <Textarea 
                  id="brute_hex" 
                  placeholder="Enter Hex data to bruteforce..."
                  className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] placeholder:text-[#52525B] min-h-[100px] font-mono text-sm" 
                  {...bruteForm.register('hex_data')} 
                />
                <p className="text-[11px] text-[#A1A1AA] mt-1">Bruteforces a single-byte XOR key.</p>
                {bruteForm.formState.errors.hex_data && (
                  <p className="text-sm text-destructive">{bruteForm.formState.errors.hex_data.message}</p>
                )}
              </div>
              <ToolActions
                isExecuting={bruteMutation.isPending}
                onExecute={() => bruteForm.handleSubmit((d) => bruteMutation.mutate(d))()}
                onClear={handleClear}
                executeLabel="Bruteforce"
              />
            </form>
          )}
        </div>
      </ToolInputPanel>

      <ToolResultPanel
        title={activeTab === 'encrypt' ? 'Encrypted Result' : activeTab === 'decrypt' ? 'Decrypted Result' : 'Bruteforce Results'}
        result={activeTab === 'encrypt' ? encryptMutation.data?.result : activeTab === 'decrypt' ? decryptMutation.data?.result : undefined}
        isLoading={isPending}
        error={activeTab === 'encrypt' ? encryptMutation.error : activeTab === 'decrypt' ? decryptMutation.error : bruteMutation.error}
        onRetry={
          activeTab === 'encrypt' ? () => encryptForm.handleSubmit((d) => encryptMutation.mutate(d))() :
          activeTab === 'decrypt' ? () => decryptForm.handleSubmit((d) => decryptMutation.mutate(d))() :
          () => bruteForm.handleSubmit((d) => bruteMutation.mutate(d))()
        }
        onClear={handleClear}
      >
        {activeTab === 'bruteforce' && bruteMutation.data && (
          <div className="space-y-2 mt-4 max-h-[400px] overflow-auto pr-2">
            {bruteMutation.data.results.map(([key, text], idx) => (
              <div key={idx} className="p-3 rounded-lg bg-[#000000] border border-[#27272A] text-sm flex gap-4">
                <div className="font-mono text-primary font-bold w-12 flex-shrink-0">
                  0x{key.toString(16).padStart(2, '0').toUpperCase()}
                </div>
                <div className="font-mono text-[#EDEDED] break-all">{text}</div>
              </div>
            ))}
          </div>
        )}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
