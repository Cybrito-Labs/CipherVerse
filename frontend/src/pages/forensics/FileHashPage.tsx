import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, Fingerprint } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToolExecution } from '@/hooks/useToolExecution';
import { CopyButton } from '@/components/shared/CopyButton';

const schema = z.object({
  filepath: z.string().min(1, 'File path is required'),
  algorithm: z.enum(['sha256', 'sha1', 'md5', 'sha512']),
});

interface FileHashResponse {
  hash: string;
}

export default function FileHashPage() {
  const mutation = useToolExecution<z.infer<typeof schema>, FileHashResponse>({ endpoint: '/file-tools/hash' });
  const form = useForm<z.infer<typeof schema>>({ resolver: zodResolver(schema), defaultValues: { filepath: '', algorithm: 'sha256' } });

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  return (
    <ToolPageLayout
      title="File Hash Calculator"
      description="Calculate cryptographic hashes for local files to verify integrity and perform forensic investigations."
      icon={Fingerprint}
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 h-fit">
          <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-4">
            <div className="space-y-2">
              <Label>Target Filepath</Label>
              <Input 
                placeholder="C:/evidence/suspect.exe" 
                className="bg-background/50 font-mono" 
                {...form.register('filepath')} 
              />
              {form.formState.errors.filepath && <p className="text-xs text-destructive">{form.formState.errors.filepath.message}</p>}
            </div>
            
            <div className="space-y-2 pt-2">
              <Label>Hash Algorithm</Label>
              <Select onValueChange={(val) => form.setValue('algorithm', val as "md5" | "sha1" | "sha256" | "sha384" | "sha512" | "sha3-256" | "sha3-512")} defaultValue={form.getValues('algorithm')}>
                <SelectTrigger className="bg-background/50">
                  <SelectValue placeholder="Algorithm" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sha256">SHA-256 (Standard)</SelectItem>
                  <SelectItem value="sha512">SHA-512</SelectItem>
                  <SelectItem value="sha1">SHA-1</SelectItem>
                  <SelectItem value="md5">MD5</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-3 pt-4">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> Calculate Hash
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
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center min-h-[300px]">
              <Fingerprint className="w-16 h-16 mb-4 opacity-20" />
              <p>Enter a file path to calculate its hash.</p>
            </div>
          ) : (
            <ResultPanel
              title="Calculated Hash"
              isLoading={mutation.isPending}
              error={mutation.error}
            >
              {res && (
                <div className="space-y-4">
                  <div className="glass rounded-xl p-6 border border-border/50 bg-background/50">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">
                        {form.getValues('algorithm')} Result
                      </h4>
                      <CopyButton text={res.hash} />
                    </div>
                    <p className="font-mono text-sm break-all leading-relaxed bg-black/20 p-4 rounded-lg border border-border/50 shadow-inner">
                      {res.hash}
                    </p>
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
