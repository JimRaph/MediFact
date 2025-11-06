

interface CacheItem<T> {
  value: T
  expires: number
}

class ClientCache {
  private cache = new Map<string, CacheItem<unknown>>()
  
  set<T>(key: string, value: T, ttl: number = 5 * 60 * 1000): void {
    const expires = Date.now() + ttl
    this.cache.set(key, { value, expires })
  }
  
  get<T>(key: string): T | null {
    const item = this.cache.get(key)
    
    if (!item) return null
    

    if (Date.now() > item.expires) {
      this.cache.delete(key)
      return null
    }
    
    return item.value as T
  }
  
  delete(key: string): void {
    this.cache.delete(key)
  }
  
  clear(): void {
    this.cache.clear()
  }
  

  cleanup(): void {
    const now = Date.now()
    for (const [key, item] of this.cache.entries()) {
      if (now > item.expires) {
        this.cache.delete(key)
      }
    }
  }
}


export const clientCache = new ClientCache()


export const cacheKeys = {
  conversations: (userId: string) => `conversations:${userId}`,
  userProfile: (userId: string) => `user:${userId}`,
  healthTopics: 'health:topics',
  frequentQuestions: 'health:frequent-questions',
} as const


export function useCache() {
  return {
    get: clientCache.get.bind(clientCache),
    set: clientCache.set.bind(clientCache),
    delete: clientCache.delete.bind(clientCache),
  }
}