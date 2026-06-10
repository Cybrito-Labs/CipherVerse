import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { ShieldCheck, Eye, EyeOff } from 'lucide-react';
import {
  ToolPageLayout,
  ToolInputPanel,
  ToolResultPanel,
  ToolActions
} from '@/components/shared/layout';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToolExecution } from '@/hooks/useToolExecution';
import { ProgressBar } from '@/components/dashboard/ProgressBar';

const schema = z.object({
  password: z.string().min(1, 'Password is required'),
});

interface PasswordResponse {
  entropy: number;
  strength: string;
}

export default function PasswordStrengthPage() {
  const [showPassword, setShowPassword] = useState(false);
  const mutation = useToolExecution<z.infer<typeof schema>, PasswordResponse>({ endpoint: '/utilities/password/strength' });
  const form = useForm<z.infer<typeof schema>>({ resolver: zodResolver(schema), defaultValues: { password: '' } });

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  const getStrengthData = (strength: string) => {
    switch (strength) {
      case 'Very Weak': return { color: 'bg-destructive', text: 'text-[#F87171]', width: 20 };
      case 'Weak': return { color: 'bg-warning', text: 'text-[#FBBF24]', width: 40 };
      case 'Reasonable': return { color: 'bg-yellow-400', text: 'text-yellow-400', width: 60 };
      case 'Strong': return { color: 'bg-success', text: 'text-[#4ADE80]', width: 80 };
      case 'Very Strong': return { color: 'bg-emerald-400', text: 'text-emerald-400', width: 100 };
      default: return { color: 'bg-muted', text: 'text-[#A1A1AA]', width: 0 };
    }
  };

  const strData = res ? getStrengthData(res.strength) : { color: 'bg-muted', text: 'text-[#A1A1AA]', width: 0 };

  return (
    <ToolPageLayout
      title="Password Strength Analyzer"
      description="Evaluate password security by calculating its information entropy (in bits). The tool provides a strength categorization and security suggestions."
      icon={ShieldCheck}
      badges={[{ label: 'Security', variant: 'success' }]}
    >
      <ToolInputPanel>
        <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-6">
          <div className="space-y-3 relative">
            <Label className="text-[#EDEDED]">Enter Password to Analyze</Label>
            <div className="relative">
              <Input
                type={showPassword ? 'text' : 'password'}
                placeholder="SuperSecretP@ssw0rd!"
                className="bg-[#000000] border-[#27272A] focus:border-[#52525B] text-[#EDEDED] placeholder:text-[#52525B] font-mono pr-10"
                {...form.register('password')}
              />
              <button
                type="button"
                className="absolute right-3 top-1/2 -translate-y-1/2 text-[#A1A1AA] hover:text-[#EDEDED]"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            {form.formState.errors.password && <p className="text-sm text-destructive">{form.formState.errors.password.message}</p>}
          </div>

          <ToolActions
            isExecuting={mutation.isPending}
            onExecute={() => form.handleSubmit((d) => mutation.mutate(d))()}
            onClear={handleClear}
            executeLabel="Evaluate Strength"
          />
        </form>
      </ToolInputPanel>

      <ToolResultPanel
        title="Security Analysis"
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit((d) => mutation.mutate(d))()}
        onClear={handleClear}
        emptyMessage="Enter a password to view its entropy and strength rating."
      >
        {res && (
          <div className="space-y-6 pt-2">
            <div>
              <div className="flex justify-between items-end mb-2">
                <span className="text-sm font-semibold uppercase tracking-wider text-[#EDEDED]">Strength Meter</span>
                <span className={`text-xl font-black uppercase ${strData.text}`}>
                  {res.strength}
                </span>
              </div>
              <ProgressBar value={strData.width} max={100} height="lg" colorClass={strData.color} />
            </div>

            <div className="p-4 rounded-xl border border-[#27272A] bg-[#000000] grid grid-cols-2 gap-4 text-center">
              <div>
                <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-1">Information Entropy</p>
                <p className="text-2xl font-mono font-bold text-[#EDEDED]">{res.entropy.toFixed(1)} <span className="text-sm opacity-50">bits</span></p>
              </div>
              <div>
                <p className="text-xs text-[#A1A1AA] uppercase tracking-wider mb-1">Complexity Rule</p>
                <p className="text-sm mt-1 text-[#EDEDED]">{res.entropy < 50 ? 'Too Simple' : res.entropy < 80 ? 'Moderate' : 'Highly Complex'}</p>
              </div>
            </div>

            <div className="p-4 rounded-xl border border-primary/20 bg-primary/5">
              <h4 className="text-sm font-semibold text-primary uppercase mb-2">Security Suggestions</h4>
              <ul className="text-sm space-y-1 list-disc list-inside text-[#A1A1AA]">
                {res.entropy < 30 && <li>Your password is extremely weak and can be cracked instantly. Add more length.</li>}
                {res.entropy < 60 && <li>Include a mix of uppercase, lowercase, numbers, and special symbols.</li>}
                {res.entropy >= 60 && res.entropy < 100 && <li>Good password. Consider making it a longer passphrase for ultimate security.</li>}
                {res.entropy >= 100 && <li>Excellent entropy. This password is safe against brute-force attacks.</li>}
                <li>Never reuse this password across multiple services.</li>
              </ul>
            </div>
          </div>
        )}
      </ToolResultPanel>
    </ToolPageLayout>
  );
}
