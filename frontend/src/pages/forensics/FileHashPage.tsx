import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Fingerprint } from 'lucide-react';
import {
  ToolPageLayout,
  ToolInputPanel,
  ToolResultPanel,
  ToolActions
} from '@/components/shared/layout';
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
      badges={[{ label: 'Forensics', variant: 'default' }, { label: 'Integrity', variant: 'success' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="space-y-3">
            <Label className="text-[#EDEDED]">Target Filepath</Label>
            <Input
              placeholder="C:/evidence/suspect.exe"
              className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] placeholder:text-[#52525B] font-mono"
              {...form.register('filepath')}
            />
            {form.formState.errors.filepath && <p className="text-sm text-destructive">{form.formState.errors.filepath.message}</p>}
          </div>

          <div className="space-y-3">
            <Label className="text-[#EDEDED]">Hash Algorithm</Label>
            <Select onValueChange={(val) => form.setValue('algorithm', val as "md5" | "sha1" | "sha256" | "sha512")} defaultValue={form.getValues('algorithm')}>
              <SelectTrigger className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED]">
                <SelectValue placeholder="Algorithm" />
              </SelectTrigger>
              <SelectContent className="bg-[#0A0A0A] border-[#27272A]">
                <SelectItem value="sha256" className="text-[#EDEDED] hover:bg-[#171717] focus:bg-[#171717]">SHA-256 (Standard)</SelectItem>
                <SelectItem value="sha512" className="text-[#EDEDED] hover:bg-[#171717] focus:bg-[#171717]">SHA-512</SelectItem>
                <SelectItem value="sha1" className="text-[#EDEDED] hover:bg-[#171717] focus:bg-[#171717]">SHA-1</SelectItem>
                <SelectItem value="md5" className="text-[#EDEDED] hover:bg-[#171717] focus:bg-[#171717]">MD5</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Calculate Hash"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Calculated Hash"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Enter a file path to calculate its hash."
      >
        {res && (
          <div className="space-y-4 pt-2">
            <div className="bg-[#000000] border border-[#27272A] rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-sm font-semibold text-primary uppercase tracking-wider">
                  {form.getValues('algorithm')} Result
                </h4>
                <CopyButton text={res.hash} />
              </div>
              <p className="font-mono text-sm break-all leading-relaxed bg-[#0A0A0A] p-4 rounded-lg border border-[#27272A] text-[#EDEDED]">
                {res.hash}
              </p>
            </div>
          </div>
        )}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
