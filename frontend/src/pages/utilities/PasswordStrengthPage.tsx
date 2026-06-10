import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, ShieldCheck, Eye, EyeOff } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
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

  // Optional: Auto-evaluate on type, but for a tool page manual submit is usually better to prevent spam.
  // We'll stick to manual submit.

  const res = mutation.data;

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  const getStrengthData = (strength: string) => {
    switch (strength) {
      case 'Very Weak': return { color: 'bg-destructive text-destructive', width: 20 };
      case 'Weak': return { color: 'bg-warning text-warning', width: 40 };
      case 'Reasonable': return { color: 'bg-yellow-400 text-yellow-400', width: 60 };
      case 'Strong': return { color: 'bg-success text-success', width: 80 };
      case 'Very Strong': return { color: 'bg-emerald-400 text-emerald-400', width: 100 };
      default: return { color: 'bg-muted text-muted-foreground', width: 0 };
    }
  };

  const strData = res ? getStrengthData(res.strength) : { color: 'bg-muted text-muted-foreground', width: 0 };

  return (
    <ToolPageLayout
      title="Password Strength Analyzer"
      description="Evaluate password security by calculating its information entropy (in bits). The tool provides a strength categorization and security suggestions."
      icon={ShieldCheck}
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 h-fit">
          <form onSubmit={form.handleSubmit((d) => mutation.mutate(d))} className="space-y-4">
            <div className="space-y-2 relative">
              <Label>Enter Password to Analyze</Label>
              <div className="relative">
                <Input 
                  type={showPassword ? 'text' : 'password'}
                  placeholder="SuperSecretP@ssw0rd!" 
                  className="bg-background/50 font-mono pr-10" 
                  {...form.register('password')} 
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {form.formState.errors.password && <p className="text-xs text-destructive">{form.formState.errors.password.message}</p>}
            </div>
            
            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> Evaluate Strength
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
              <ShieldCheck className="w-16 h-16 mb-4 opacity-20" />
              <p>Enter a password to view its entropy and strength rating.</p>
            </div>
          ) : (
            <ResultPanel
              title="Security Analysis"
              isLoading={mutation.isPending}
              error={mutation.error}
            >
              {res && (
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between items-end mb-2">
                      <span className="text-sm font-semibold uppercase tracking-wider">Strength Meter</span>
                      <span className={`text-xl font-black uppercase ${strData.color.split(' ')[1]}`}>
                        {res.strength}
                      </span>
                    </div>
                    {/* Reusing ProgressBar but explicitly passing the width for the strength tiers instead of maxing to a set value */}
                    <ProgressBar value={strData.width} max={100} height="lg" colorClass={strData.color.split(' ')[0]} />
                  </div>

                  <div className="p-4 rounded-xl border border-border bg-background/30 grid grid-cols-2 gap-4 text-center">
                    <div>
                      <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Information Entropy</p>
                      <p className="text-2xl font-mono font-bold">{res.entropy.toFixed(1)} <span className="text-sm opacity-50">bits</span></p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Complexity Rule</p>
                      <p className="text-sm mt-1">{res.entropy < 50 ? 'Too Simple' : res.entropy < 80 ? 'Moderate' : 'Highly Complex'}</p>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl border border-primary/20 bg-primary/5">
                    <h4 className="text-sm font-semibold text-primary uppercase mb-2">Security Suggestions</h4>
                    <ul className="text-sm space-y-1 list-disc list-inside text-muted-foreground">
                      {res.entropy < 30 && <li>Your password is extremely weak and can be cracked instantly. Add more length.</li>}
                      {res.entropy < 60 && <li>Include a mix of uppercase, lowercase, numbers, and special symbols.</li>}
                      {res.entropy >= 60 && res.entropy < 100 && <li>Good password. Consider making it a longer passphrase for ultimate security.</li>}
                      {res.entropy >= 100 && <li>Excellent entropy. This password is safe against brute-force attacks.</li>}
                      <li>Never reuse this password across multiple services.</li>
                    </ul>
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
