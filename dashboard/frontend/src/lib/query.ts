import { QueryClient } from "@tanstack/react-query"

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,       // 1 min before refetch
      gcTime: 5 * 60_000,     // 5 min garbage collection
      retry: (failureCount, error) => {
        // Don't retry on "warming up" errors - they'll resolve when jobs finish
        if (error instanceof Error && error.message === "warming_up") return false
        return failureCount < 2
      },
      refetchOnWindowFocus: false,
    },
  },
})
