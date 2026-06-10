import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import type { z, ZodObject, ZodRawShape } from 'zod';
import { motion } from 'framer-motion';
import { Settings2, ChevronDown, ChevronUp } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
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
  badges?: { label: string; variant?: 'default' | 'success' | 'warning' }[];
  hasTabs?: boolean;
  encryptEndpoint?: string;
  decryptEndpoint?: string;
  encryptLabel?: string;
  decryptLabel?: string;
  transformPayload?: (data: any, tab: 'encrypt' | 'decrypt') => any;
}

export default function CipherToolPage({
  title,
  description,
  icon,
  badges,
  fields,
  endpoint,
  schema,
  hasTabs = false,
  encryptEndpoint,
  decryptEndpoint,
  encryptLabel = 'Encrypt',
  decryptLabel = 'Decrypt',
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
        <Label htmlFor={field.name} className="text-foreground">
          {field.label}
        </Label>
      )}
      {field.type === 'textarea' ? (
        <Textarea
          id={field.name}
          placeholder={field.placeholder}
          className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground min-h-[100px] font-mono text-sm"
          {...form.register(field.name)}
        />
      ) : field.type === 'number' ? (
        <Input
          id={field.name}
          type="number"
          placeholder={field.placeholder}
          className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground font-mono"
          {...form.register(field.name, { valueAsNumber: true })}
        />
      ) : field.type === 'select' ? (
        <Select
          onValueChange={(value) => form.setValue(field.name, value)}
          defaultValue={field.defaultValue?.toString()}
        >
          <SelectTrigger className="bg-background border-border focus:border-muted-foreground text-foreground">
            <SelectValue placeholder={field.placeholder} />
          </SelectTrigger>
          <SelectContent className="bg-card border-border">
            {field.options?.map((opt) => (
              <SelectItem key={opt.value} value={opt.value} className="text-foreground hover:bg-secondary focus:bg-secondary">
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
          <Label htmlFor={field.name} className="text-sm font-medium cursor-pointer text-foreground">
            {field.label}
          </Label>
        </div>
      ) : (
        <Input
          id={field.name}
          type="text"
          placeholder={field.placeholder}
          className="bg-background border-border focus:border-muted-foreground text-foreground placeholder:text-muted-foreground"
          {...form.register(field.name)}
        />
      )}
      {field.description && field.type !== 'checkbox' && (
        <p className="text-[11px] text-muted-foreground">{field.description}</p>
      )}
      {field.description && field.type === 'checkbox' && (
        <p className="text-[11px] text-muted-foreground ml-6">{field.description}</p>
      )}
      {form.formState.errors[field.name] && (
        <p className="text-sm text-destructive">
          {form.formState.errors[field.name]?.message as string}
        </p>
      )}
    </div>
  );

  const regularFields = fields.filter(f => !f.isAdvanced);
  const advancedFields = fields.filter(f => f.isAdvanced);

  const renderForm = () => (
    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
      {regularFields.map(renderField)}

      {advancedFields.length > 0 && (
        <div className="pt-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs text-muted-foreground hover:text-foreground hover:bg-secondary p-2 h-auto"
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
            <div className="space-y-6 pt-4 border-t border-border mt-3">
              {advancedFields.map(renderField)}
            </div>
          </motion.div>
        </div>
      )}

      <ToolActions
        isExecuting={mutation.isPending}
        onExecute={() => form.handleSubmit(onSubmit)()}
        onClear={handleClear}
        executeLabel={hasTabs ? (activeTab === 'encrypt' ? encryptLabel : decryptLabel) : 'Execute'}
      />
    </form>
  );

  return (
    <ToolPageLayout title={title} description={description} icon={icon} badges={badges}>
      <ToolInputPanel>
        {hasTabs ? (
          <div className="flex flex-col gap-6">
            <ToolTabs
              tabs={[
                { id: 'encrypt', label: encryptLabel },
                { id: 'decrypt', label: decryptLabel }
              ]}
              activeTab={activeTab}
              onTabChange={(tab) => {
                setActiveTab(tab as 'encrypt' | 'decrypt');
                mutation.reset();
              }}
            />
            {renderForm()}
          </div>
        ) : (
          renderForm()
        )}
      </ToolInputPanel>

      <ToolResultPanel
        title={hasTabs ? (activeTab === 'encrypt' ? `${encryptLabel}ed Result` : `${decryptLabel}ed Result`) : 'Result'}
        result={mutation.data?.result}
        isLoading={mutation.isPending}
        error={mutation.error}
        onRetry={() => form.handleSubmit(onSubmit)()}
        onClear={() => mutation.reset()}
      />
    </ToolPageLayout>
  );
}
