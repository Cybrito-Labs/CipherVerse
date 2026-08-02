import { lazy, Suspense } from 'react';
import { RouterProvider } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Analytics } from '@vercel/analytics/react';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Toaster } from '@/components/ui/sonner';
import { router } from '@/routes';
import { ThemeProvider } from '@/components/ThemeProvider';

const ChatbotWidget = lazy(() =>
  import('@/components/shared/ChatbotWidget').then((m) => ({ default: m.ChatbotWidget }))
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000,
    },
    mutations: {
      retry: 0,
    },
  },
});

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider delayDuration={200}>
          <RouterProvider router={router} />
          <Suspense fallback={null}>
            <ChatbotWidget />
          </Suspense>
          <Analytics />
          <Toaster
            position="bottom-right"
            toastOptions={{
              className: 'glass border-border',
            }}
          />
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
}

export default App;
