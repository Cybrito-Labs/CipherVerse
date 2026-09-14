import { useRouteError, isRouteErrorResponse } from 'react-router-dom';
import { RefreshCw, AlertTriangle, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function RouteErrorBoundary() {
  const error = useRouteError();
  const errorMessage = error instanceof Error ? error.message : '';
  const isChunkError =
    errorMessage.includes('dynamically imported module') ||
    errorMessage.includes('Loading chunk') ||
    errorMessage.includes('Failed to fetch') ||
    errorMessage.includes('Importing a module script failed');

  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center p-6 text-center">
      <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400 mb-4 shadow-sm">
        <AlertTriangle className="w-7 h-7" />
      </div>
      <h2 className="text-xl font-semibold text-foreground mb-2">
        {isChunkError ? 'App Update or Session Reconnected' : 'Something went wrong'}
      </h2>
      <p className="text-sm text-muted-foreground max-w-md mb-6 leading-relaxed">
        {isChunkError
          ? 'A new version of CipherVerse was compiled or your connection refreshed. Reload the application to get the latest tools and scripts.'
          : isRouteErrorResponse(error)
          ? `${error.status}: ${error.statusText || 'Page failed to load'}`
          : 'An unexpected application error occurred while rendering this view.'}
      </p>
      <div className="flex items-center gap-3">
        <Button
          onClick={() => window.location.reload()}
          className="gap-2 shadow-sm"
        >
          <RefreshCw className="w-4 h-4" />
          Reload Application
        </Button>
        <Button
          variant="outline"
          onClick={() => (window.location.href = '/')}
          className="gap-2 border-border"
        >
          <Home className="w-4 h-4" />
          Go to Home
        </Button>
      </div>
    </div>
  );
}
