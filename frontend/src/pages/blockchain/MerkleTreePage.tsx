import { useForm, useFieldArray } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, Network, Plus, Trash2 } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToolExecution } from '@/hooks/useToolExecution';
import { ResultPanel } from '@/components/shared/ResultPanel';

const schema = z.object({
  items: z.array(z.object({ value: z.string().min(1, 'Cannot be empty') })).min(1, 'At least one item is required'),
  algorithm: z.enum(['sha256', 'sha1', 'md5']),
});

interface MerkleResponse {
  root: string;
  tree: Record<string, string[]>;
}

export default function MerkleTreePage() {
  const mutation = useToolExecution<z.infer<typeof schema>, MerkleResponse>({ endpoint: '/blockchain/merkle' });
  const form = useForm<z.infer<typeof schema>>({ 
    resolver: zodResolver(schema), 
    defaultValues: { items: [{ value: 'Item 1' }, { value: 'Item 2' }], algorithm: 'sha256' } 
  });
  
  const { fields, append, remove } = useFieldArray({
    control: form.control,
    name: "items"
  });

  const handleClear = () => {
    form.reset({ items: [{ value: '' }], algorithm: 'sha256' });
    mutation.reset();
  };

  const onSubmit = (data: z.infer<typeof schema>) => {
    mutation.mutate({
      items: data.items.map(i => i.value),
      algorithm: data.algorithm
    });
  };

  const res = mutation.data;

  // Convert dictionary keys to numbers and sort descending for top-down rendering
  const levels = res ? Object.keys(res.tree).map(Number).sort((a, b) => b - a) : [];

  return (
    <ToolPageLayout
      title="Merkle Tree Builder"
      description="Construct and visualize cryptographic Merkle trees. Enter a list of items to generate their leaves and compute the recursive hashes up to the Merkle Root."
      icon={Network}
    >
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <motion.div initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} className="glass rounded-xl p-6 lg:col-span-4 h-fit">
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Data Items (Leaves)</Label>
                <Button type="button" variant="outline" size="sm" onClick={() => append({ value: '' })}>
                  <Plus className="w-4 h-4 mr-1" /> Add
                </Button>
              </div>
              
              <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
                {fields.map((field, index) => (
                  <div key={field.id} className="flex items-center gap-2">
                    <Input 
                      placeholder={`Data Item ${index + 1}`} 
                      className="bg-background/50" 
                      {...form.register(`items.${index}.value`)} 
                    />
                    <Button 
                      type="button" 
                      variant="ghost" 
                      size="icon" 
                      className="text-destructive hover:bg-destructive/20"
                      onClick={() => remove(index)}
                      disabled={fields.length === 1}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>
              {form.formState.errors.items && <p className="text-xs text-destructive">{form.formState.errors.items.message}</p>}
            </div>

            <div className="space-y-2 pt-4 border-t border-border">
              <Label>Hash Algorithm</Label>
              <Select onValueChange={(val) => form.setValue('algorithm', val as "sha256" | "sha1" | "md5")} defaultValue={form.getValues('algorithm')}>
                <SelectTrigger className="bg-background/50">
                  <SelectValue placeholder="Algorithm" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sha256">SHA-256 (Bitcoin standard)</SelectItem>
                  <SelectItem value="sha1">SHA-1</SelectItem>
                  <SelectItem value="md5">MD5</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex items-center gap-3 pt-2">
              <Button type="submit" disabled={mutation.isPending} className="w-full bg-primary text-primary-foreground gap-2">
                <Play className="w-4 h-4"/> Build Tree
              </Button>
              <Button type="button" variant="outline" onClick={handleClear} className="px-3">
                <RotateCcw className="w-4 h-4" />
              </Button>
            </div>
          </form>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="lg:col-span-8">
          {!res && !mutation.isPending && !mutation.error ? (
            <div className="glass rounded-xl p-12 text-center text-muted-foreground flex flex-col items-center justify-center min-h-[500px]">
              <Network className="w-16 h-16 mb-4 opacity-20" />
              <p>Add data items and build to visualize the Merkle Tree.</p>
            </div>
          ) : (
            <ResultPanel
              title="Merkle Tree Visualization"
              isLoading={mutation.isPending}
              error={mutation.error}
            >
              {res && (
                <div className="space-y-8 overflow-x-auto pb-4">
                  <div className="p-4 rounded-xl border bg-primary/10 border-primary/30 text-center">
                    <p className="text-xs font-semibold text-primary uppercase mb-1">Merkle Root</p>
                    <p className="font-mono text-sm break-all font-bold">{res.root}</p>
                  </div>
                  
                  <div className="flex flex-col items-center gap-8 relative">
                    {levels.map((levelIndex) => (
                      <div key={levelIndex} className="flex flex-col items-center w-full">
                        <p className="text-[10px] uppercase text-muted-foreground font-bold mb-2 tracking-widest">
                          {levelIndex === levels[0] ? 'Root Level' : levelIndex === 0 ? 'Leaves (Hashed)' : `Level ${levelIndex}`}
                        </p>
                        <div className="flex justify-center flex-wrap gap-4 w-full">
                          {res.tree[levelIndex].map((node, i) => (
                            <div key={i} className="flex flex-col items-center relative">
                              <div className="px-3 py-2 rounded-lg border bg-background text-[10px] font-mono shadow-sm truncate max-w-[150px] sm:max-w-[200px]" title={node}>
                                {node.substring(0, 16)}...{node.substring(node.length - 8)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
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
