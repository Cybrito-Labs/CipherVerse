import { useMutation } from '@tanstack/react-query';
import { api } from '@/api/client';

interface UseToolExecutionOptions<TRequest, TResponse> {
  endpoint: string;
  method?: 'post' | 'get' | 'getWithParams';
  onSuccess?: (data: TResponse) => void;
  onError?: (error: Error) => void;
}

export function useToolExecution<TRequest = unknown, TResponse = unknown>({
  endpoint,
  method = 'post',
  onSuccess,
  onError,
}: UseToolExecutionOptions<TRequest, TResponse>) {
  return useMutation<TResponse, Error, TRequest>({
    mutationFn: async (data: TRequest) => {
      switch (method) {
        case 'post':
          return api.post<TResponse>(endpoint, data);
        case 'get':
          return api.get<TResponse>(endpoint);
        case 'getWithParams':
          return api.getWithParams<TResponse>(endpoint, data as Record<string, unknown>);
        default:
          return api.post<TResponse>(endpoint, data);
      }
    },
    onSuccess,
    onError,
  });
}
