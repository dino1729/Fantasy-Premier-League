"""Cache Manager Module

Provides intelligent caching for FPL API responses to speed up development.

This implementation stores all cache entries in a single bundled pickle file
to avoid creating dozens of per-key pickle files on disk.
"""

import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
import functools


class CacheManager:
    """Manages caching of API responses and computed data."""
    
    # Default TTL values (in seconds)
    DEFAULT_TTL = {
        'bootstrap': 3600,        # 1 hour - static data changes infrequently
        'fixtures': 3600,         # 1 hour
        'team_data': 300,         # 5 minutes - team data changes more often
        'player_history': 3600,   # 1 hour
        'gw_picks': 300,          # 5 minutes
        'competitive': 300,       # 5 minutes
        'predictions': 1800,      # 30 minutes
        'league_standings': 300,  # 5 minutes - league standings
        'league_ownership': 300,  # 5 minutes - league ownership data
        'default': 600            # 10 minutes
    }
    
    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to ./cache
            enabled: Whether caching is enabled. Set to False to disable all caching.
        """
        self.enabled = enabled
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache"
        
        self.cache_dir = Path(cache_dir)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Single bundled cache file (avoid per-key .pkl fan-out)
        self.cache_file = self.cache_dir / "cache_bundle.pkl"

        if self.enabled:
            self._cleanup_legacy_artifacts()

        # In-memory structures loaded from bundle
        bundle = self._load_bundle()
        self._data: Dict[str, Any] = bundle.get("data", {})
        self.metadata: Dict[str, Dict[str, Any]] = bundle.get("metadata", {})

    def _cleanup_legacy_artifacts(self):
        """Remove legacy per-key pickle files and metadata json.

        Keep known session cache files if present.
        """
        # Remove legacy json metadata used by older CacheManager implementation
        meta_path = self.cache_dir / "cache_metadata.json"
        if meta_path.exists():
            try:
                meta_path.unlink()
            except Exception:
                pass

        # Remove per-key md5-named cache pickles from the old implementation.
        # Preserve:
        # - bundled cache file used by this CacheManager
        # - session cache pickles used by SessionCacheManager
        keep_names = {"cache_bundle.pkl", "session_cache.pkl"}
        for pkl_file in self.cache_dir.glob("*.pkl"):
            if pkl_file.name in keep_names or pkl_file.name.startswith("session_"):
                continue
            try:
                pkl_file.unlink()
            except Exception:
                pass
    
    def _load_bundle(self) -> Dict[str, Any]:
        """Load cache bundle from disk."""
        if not self.enabled or not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, "rb") as f:
                loaded = pickle.load(f)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    def _save_bundle(self):
        """Persist cache bundle to disk (single file)."""
        if not self.enabled:
            return

        try:
            bundle = {"data": self._data, "metadata": self.metadata}
            tmp_path = self.cache_file.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(bundle, f)
            tmp_path.replace(self.cache_file)
        except Exception as e:
            print(f"[WARN] Failed to save cache bundle: {e}")
    
    def _generate_key(self, cache_type: str, *args, **kwargs) -> str:
        """Generate a unique cache key from function arguments.
        
        Args:
            cache_type: Type of cache (e.g., 'bootstrap', 'team_data')
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key
        
        Returns:
            Unique cache key as hex string
        """
        # Create a string representation of all arguments
        key_parts = [cache_type] + [str(arg) for arg in args]
        
        # Add sorted kwargs
        for k in sorted(kwargs.keys()):
            key_parts.append(f"{k}={kwargs[k]}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_expired(self, key: str, ttl: int) -> bool:
        """Check if a cache entry is expired.
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds
        
        Returns:
            True if expired or not found, False otherwise
        """
        if key not in self.metadata:
            return True
        
        cached_time = datetime.fromisoformat(self.metadata[key]['timestamp'])
        expiry_time = cached_time + timedelta(seconds=ttl)
        return datetime.now() > expiry_time
    
    def get(self, cache_type: str, *args, **kwargs) -> Optional[Any]:
        """Retrieve data from cache.
        
        Args:
            cache_type: Type of cache
            *args: Arguments that identify the cached data
            **kwargs: Additional arguments
        
        Returns:
            Cached data if found and valid, None otherwise
        """
        if not self.enabled:
            return None
        
        key = self._generate_key(cache_type, *args, **kwargs)

        if key not in self._data:
            return None
        
        # Check if expired
        ttl = self.DEFAULT_TTL.get(cache_type, self.DEFAULT_TTL['default'])
        if self._is_expired(key, ttl):
            # Clean up expired cache
            try:
                del self._data[key]
                del self.metadata[key]
                self._save_bundle()
            except Exception:
                pass
            return None
        
        # Load and return cached data
        try:
            return self._data.get(key)
        except Exception as e:
            print(f"[WARN] Failed to load cache for {cache_type}: {e}")
            return None
    
    def set(self, cache_type: str, data: Any, *args, **kwargs):
        """Store data in cache.
        
        Args:
            cache_type: Type of cache
            data: Data to cache
            *args: Arguments that identify the cached data
            **kwargs: Additional arguments
        """
        if not self.enabled:
            return
        
        key = self._generate_key(cache_type, *args, **kwargs)

        # Save data into bundle
        try:
            self._data[key] = data
            
            # Update metadata
            self.metadata[key] = {
                'cache_type': cache_type,
                'timestamp': datetime.now().isoformat(),
                'args': args,
                'kwargs': kwargs
            }
            self._save_bundle()
        except Exception as e:
            print(f"[WARN] Failed to cache {cache_type}: {e}")
    
    def invalidate(self, cache_type: Optional[str] = None):
        """Invalidate cache entries.
        
        Args:
            cache_type: If specified, only invalidate entries of this type.
                       If None, invalidate all cache.
        """
        if not self.enabled:
            return
        
        if cache_type is None:
            # Clear all cache
            try:
                self._data.clear()
                self.metadata.clear()
                if self.cache_file.exists():
                    try:
                        self.cache_file.unlink()
                    except Exception:
                        pass
                print("[INFO] All cache cleared")
            except Exception as e:
                print(f"[WARN] Failed to clear cache: {e}")
        else:
            # Clear specific cache type
            keys_to_remove = [
                key for key, meta in self.metadata.items()
                if meta.get('cache_type') == cache_type
            ]
            
            for key in keys_to_remove:
                try:
                    if key in self._data:
                        del self._data[key]
                    del self.metadata[key]
                except Exception:
                    pass
            
            self._save_bundle()
            print(f"[INFO] Cache cleared for type: {cache_type}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {'enabled': False}
        
        stats = {
            'enabled': True,
            'total_entries': len(self.metadata),
            'cache_dir': str(self.cache_dir),
            'by_type': {}
        }
        
        for key, meta in self.metadata.items():
            cache_type = meta.get('cache_type', 'unknown')
            if cache_type not in stats['by_type']:
                stats['by_type'][cache_type] = 0
            stats['by_type'][cache_type] += 1
        
        return stats


def cached(cache_type: str, cache_manager_attr: str = 'cache'):
    """Decorator to cache method results.
    
    Usage:
        class MyClass:
            def __init__(self):
                self.cache = CacheManager()
            
            @cached('my_data')
            def get_data(self, arg1, arg2):
                # expensive operation
                return data
    
    Args:
        cache_type: Type of cache for TTL lookup
        cache_manager_attr: Attribute name of the CacheManager instance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            cache_mgr: CacheManager = getattr(self, cache_manager_attr)
            
            # Try to get from cache
            cached_data = cache_mgr.get(cache_type, *args, **kwargs)
            if cached_data is not None:
                return cached_data
            
            # Call original function
            result = func(self, *args, **kwargs)
            
            # Cache the result
            cache_mgr.set(cache_type, result, *args, **kwargs)
            
            return result
        
        return wrapper
    return decorator

