import axios, { type AxiosInstance, type AxiosError, type InternalAxiosRequestConfig, type AxiosResponse } from 'axios';
import { toast } from 'sonner';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    if (import.meta.env.DEV) {
      console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data || '');
    }
    return config;
  },
  (error: AxiosError) => {
    console.error('[API] Request error:', error.message);
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error: AxiosError<{ detail?: string }>) => {
    const message = error.response?.data?.detail
      || error.response?.statusText
      || error.message
      || 'An unexpected error occurred';

    const status = error.response?.status;

    if (status === 422) {
      toast.error('Validation Error', { description: message });
    } else if (status === 404) {
      toast.error('Not Found', { description: 'The requested endpoint was not found.' });
    } else if (status === 500) {
      toast.error('Server Error', { description: 'An internal server error occurred.' });
    } else if (error.code === 'ECONNABORTED') {
      toast.error('Timeout', { description: 'The request timed out. Please try again.' });
    } else if (!error.response) {
      toast.error('Network Error', { description: 'Unable to connect to the server.' });
    } else {
      toast.error('Error', { description: message });
    }

    return Promise.reject(error);
  }
);

export const api = {
  post: <T>(url: string, data?: unknown) =>
    apiClient.post<T>(url, data).then((res) => res.data),

  get: <T>(url: string) =>
    apiClient.get<T>(url).then((res) => res.data),

  getWithParams: <T>(url: string, params: Record<string, unknown>) =>
    apiClient.get<T>(url, { params }).then((res) => res.data),
};

export default apiClient;
