import { motion } from 'framer-motion';

export function LoadingState() {
  return (
    <div className="space-y-4">
      {/* Vercel-style skeleton loaders */}
      <div className="flex items-center justify-between">
        <div className="h-4 w-32 rounded bg-[#171717] skeleton-shimmer"></div>
        <div className="h-8 w-24 rounded bg-[#171717] skeleton-shimmer"></div>
      </div>
      
      <div className="space-y-2 mt-4">
        <div className="h-24 w-full rounded-lg bg-[#171717] skeleton-shimmer"></div>
        <div className="flex gap-2">
          <div className="h-16 w-1/2 rounded-lg bg-[#171717] skeleton-shimmer"></div>
          <div className="h-16 w-1/2 rounded-lg bg-[#171717] skeleton-shimmer"></div>
        </div>
      </div>
    </div>
  );
}
