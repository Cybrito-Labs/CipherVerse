import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import type { z, ZodObject, ZodRawShape } from 'zod';
import { motion } from 'framer-motion';
import { Play, RotateCcw, Settings2, ChevronDown, ChevronUp } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { ToolPageLayout } from '@/components/shared/ToolPageLayout';
import { ResultPanel } from '@/components/shared/ResultPanel';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToolExecution } from '@/hooks/useToolExecution';
import type { ClassicalResponse } from '@/types/api';

export interface FieldConfig {
  name: string;
  label: string;
  type: 'text' | 'textarea' | 'number' | 'select' | 'checkbox';
  placeholder?: string;
  defaultValue?: string | number;
  description?: string;
  isAdvanced?: boolean;
  options?: { label: string; value: string }[];
}

interface CipherToolPageProps {
  title: string;
  description: string;
  icon: LucideIcon;
  fields: FieldConfig[];
  endpoint: string;
  schema: ZodObject<ZodRawShape>;
  hasTabs?: boolean;
  encryptEndpoint?: string;
  decryptEndpoint?: string;
  transformPayload?: (data: Record<string, unknown>, tab: 'encrypt' | 'decrypt') => Record<string, unknown>;
}

export default function CipherToolPage({
  title,
  description,
  icon,
  fields,
  endpoint,
  schema,
  hasTabs = false,
  encryptEndpoint,
  decryptEndpoint,
  transformPayload,
}: CipherToolPageProps) {
  const [activeTab, setActiveTab] = useState<'encrypt' | 'decrypt'>('encrypt');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const currentEndpoint = hasTabs
    ? activeTab === 'encrypt'
      ? encryptEndpoint!
      : decryptEndpoint!
    : endpoint;

  const mutation = useToolExecution<z.infer<typeof schema>, ClassicalResponse>({
    endpoint: currentEndpoint,
  });

  const form = useForm<z.infer<typeof schema>>({
    resolver: zodResolver(schema),
    defaultValues: fields.reduce(
      (acc, f) => ({
        ...acc,
        [f.name]: f.defaultValue ?? (f.type === 'number' ? 0 : ''),
      }),
      {} as Record<string, unknown>
    ),
  });

  const onSubmit = (data: z.infer<typeof schema>) => {
    const payload = transformPayload ? transformPayload(data, activeTab) : data;
    mutation.mutate(payload);
  };

  const handleClear = () => {
    form.reset();
    mutation.reset();
  };

  const renderField = (field: FieldConfig) => (
    <div key={field.name} className={`space-y-2 ${field.type === 'checkbox' ? 'flex flex-row items-center gap-2 space-y-0' : ''}`}>
      {field.type !== 'checkbox' && (
        <Label htmlFor={field.name} className="text-sm font-medium">
          {field.label}
        </Label>
      )}
      {field.type === 'textarea' ? (
        <Textarea
          id={field.name}
          placeholder={field.placeholder}
          className="bg-background/50 border-border min-h-[100px] font-mono text-sm"
          {...form.register(field.name)}
        />
      ) : field.type === 'number' ? (
        <Input
          id={field.name}
          type="number"
          placeholder={field.placeholder}
          className="bg-background/50 border-border font-mono"
          {...form.register(field.name, { valueAsNumber: true })}
        />
      ) : field.type === 'select' ? (
        <Select
          onValueChange={(value) => form.setValue(field.name, value)}
          defaultValue={field.defaultValue?.toString()}
        >
          <SelectTrigger className="bg-background/50 border-border">
            <SelectValue placeholder={field.placeholder} />
          </SelectTrigger>
          <SelectContent>
            {field.options?.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : field.type === 'checkbox' ? (
        <div className="flex items-center gap-2 mt-2">
          <input
            id={field.name}
            type="checkbox"
            className="w-4 h-4 rounded border-border bg-background text-primary focus:ring-primary"
            {...form.register(field.name)}
          />
          <Label htmlFor={field.name} className="text-sm font-medium cursor-pointer">
            {field.label}
          </Label>
        </div>
      ) : (
        <Input
          id={field.name}
          type="text"
          placeholder={field.placeholder}
          className="bg-background/50 border-border"
          {...form.register(field.name)}
        />
      )}
      {field.description && field.type !== 'checkbox' && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
      {field.description && field.type === 'checkbox' && (
        <p className="text-xs text-muted-foreground ml-6">{field.description}</p>
      )}
      {form.formState.errors[field.name] && (
        <p className="text-xs text-destructive">
          {form.formState.errors[field.name]?.message as string}
        </p>
      )}
    </div>
  );

  const regularFields = fields.filter(f => !f.isAdvanced);
  const advancedFields = fields.filter(f => f.isAdvanced);

  const renderForm = () => (
    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
      {regularFields.map(renderField)}

      {advancedFields.length > 0 && (
        <div className="pt-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs text-muted-foreground hover:text-foreground p-0 h-auto"
          >
            <Settings2 className="w-3.5 h-3.5 mr-1.5" />
            Advanced Settings
            {showAdvanced ? (
              <ChevronUp className="w-3.5 h-3.5 ml-1" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5 ml-1" />
            )}
          </Button>
          
          <motion.div
            initial={false}
            animate={{ height: showAdvanced ? 'auto' : 0, opacity: showAdvanced ? 1 : 0 }}
            className="overflow-hidden"
          >
            <div className="space-y-4 pt-4 border-t border-border mt-3">
              {advancedFields.map(renderField)}
            </div>
          </motion.div>
        </div>
      )}

      <div className="flex items-center gap-3 pt-2">
        <Button
          type="submit"
          disabled={mutation.isPending}
          className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
        >
          <Play className="w-4 h-4" />
          {mutation.isPending ? 'Executing...' : hasTabs ? (activeTab === 'encrypt' ? 'Encrypt' : 'Decrypt') : 'Execute'}
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={handleClear}
          className="gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Clear
        </Button>
      </div>
    </form>
  );

  return (
    <ToolPageLayout title={title} description={description} icon={icon}>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Section */}
        <motion.div
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
          className="glass rounded-xl p-6"
        >
          {hasTabs ? (
            <Tabs
              value={activeTab}
              onValueChange={(v) => {
                setActiveTab(v as 'encrypt' | 'decrypt');
                mutation.reset();
              }}
            >
              <TabsList className="mb-4 bg-background/50">
                <TabsTrigger value="encrypt">Encrypt</TabsTrigger>
                <TabsTrigger value="decrypt">Decrypt</TabsTrigger>
              </TabsList>
              <TabsContent value="encrypt">{renderForm()}</TabsContent>
              <TabsContent value="decrypt">{renderForm()}</TabsContent>
            </Tabs>
          ) : (
            renderForm()
          )}
        </motion.div>

        {/* Result Section */}
        <motion.div
          initial={{ opacity: 0, x: 12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <ResultPanel
            title={hasTabs ? (activeTab === 'encrypt' ? 'Encrypted Result' : 'Decrypted Result') : 'Result'}
            result={mutation.data?.result}
            isLoading={mutation.isPending}
            error={mutation.error}
            onRetry={() => form.handleSubmit(onSubmit)()}
          />
          {!mutation.data && !mutation.isPending && !mutation.error && (
            <div className="glass rounded-xl p-12 text-center">
              <p className="text-sm text-muted-foreground">
                Enter your data and click Execute to see results
              </p>
            </div>
          )}
        </motion.div>
      </div>
    </ToolPageLayout>
  );
}
