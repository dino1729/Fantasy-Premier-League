"""Session-Based Cache Manager

Provides session-based caching where all data for a single report run
is bundled into one cache file. This simplifies cache management and
reduces file clutter compared to granular per-item caching.

Usage:
    cache = SessionCacheManager(team_id=847569, gameweek=17)
    
    # During session
    data = cache.get('bootstrap')
    if data is None:
        data = fetch_data()
        cache.set('bootstrap', data)
    
    # At end of session
    cache.save()
"""

import pickle
import json
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timedelta


class SessionCacheManager:
    """Manages session-based caching for FPL reports.
    
    All cached data for a single report run is stored in one session file.
    Sessions are identified by team_id + gameweek + date.
    """
    
    def __init__(
        self, 
        team_id: int,
        gameweek: int,
        cache_dir: Optional[Path] = None,
        ttl: int = 3600,
        max_sessions: int = 10,
        enabled: bool = True,
        single_file: bool = False
    ):
        """Initialize session cache manager.
        
        Args:
            team_id: FPL team ID for this session.
            gameweek: Gameweek number for this session.
            cache_dir: Directory to store cache files. Defaults to ./cache
            ttl: Time-to-live in seconds for session files (default: 1 hour).
            max_sessions: Maximum number of session files to keep (default: 10).
            enabled: Whether caching is enabled. Set to False to disable all caching.
        """
        self.enabled = enabled
        self.single_file = single_file
        self.team_id = team_id
        self.gameweek = gameweek
        self.ttl = ttl
        self.max_sessions = max_sessions
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache"
        self.cache_dir = Path(cache_dir)
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate session ID (includes date for uniqueness unless single-file mode)
        if self.single_file:
            self.session_id = "session_cache"
        else:
            date_str = datetime.now().strftime('%Y%m%d')
            self.session_id = f"session_{team_id}_gw{gameweek}_{date_str}"
        
        # In-memory cache for current session
        self.cache_data: Dict[str, Any] = {}
        self.session_metadata = {
            'team_id': team_id,
            'gameweek': gameweek,
            'created': datetime.now().isoformat(),
            'ttl': ttl
        }
        
        # Session file paths
        if self.single_file:
            # Fixed filename - overwritten per run to avoid file buildup
            self.session_file = self.cache_dir / "session_cache.pkl"
            self.metadata_file = None
        else:
            self.session_file = self.cache_dir / f"{self.session_id}.pkl"
            self.metadata_file = self.cache_dir / "session_metadata.json"
        
        # Try to load existing session
        if self.enabled:
            if self.single_file:
                self._cleanup_legacy_artifacts_single_file()
            self._load_session()
            if not self.single_file:
                self._cleanup_expired()

    def _cleanup_legacy_artifacts_single_file(self):
        """Delete legacy cache artifacts so only a single bundled file remains.

        In single-file mode we want exactly one cache pickle file on disk:
        - keep: session_cache.pkl
        - delete: md5-named CacheManager .pkl files, session_*.pkl files, and legacy json metadata
        """
        if not self.enabled:
            return

        # Delete any other pickle files in the cache dir
        for pkl_file in self.cache_dir.glob("*.pkl"):
            if pkl_file.name == "session_cache.pkl":
                continue
            try:
                pkl_file.unlink()
            except Exception:
                pass

        # Delete legacy metadata files
        for meta_name in ("cache_metadata.json", "session_metadata.json"):
            meta_path = self.cache_dir / meta_name
            if meta_path.exists():
                try:
                    meta_path.unlink()
                except Exception:
                    pass
    
    def _generate_cache_key(self, cache_type: str, *args, **kwargs) -> str:
        """Generate a unique cache key within the session.
        
        Args:
            cache_type: Type of cache (e.g., 'bootstrap', 'team_data')
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key
        
        Returns:
            Unique cache key as string
        """
        # Create a string representation of all arguments
        key_parts = [cache_type] + [str(arg) for arg in args]
        
        # Add sorted kwargs
        for k in sorted(kwargs.keys()):
            key_parts.append(f"{k}={kwargs[k]}")
        
        key_string = "|".join(key_parts)
        return key_string
    
    def _load_session(self):
        """Load existing session from disk if available and not expired."""
        if not self.session_file.exists():
            return
        
        try:
            # Check if session is expired
            file_mtime = datetime.fromtimestamp(self.session_file.stat().st_mtime)
            age = datetime.now() - file_mtime
            
            if age.total_seconds() > self.ttl:
                # Session expired, delete it
                self.session_file.unlink()
                return
            
            # Load session data
            with open(self.session_file, 'rb') as f:
                loaded_data = pickle.load(f)
                self.cache_data = loaded_data.get('cache_data', {})
                self.session_metadata = loaded_data.get('metadata', self.session_metadata)
            
        except Exception as e:
            print(f"[WARN] Failed to load session {self.session_id}: {e}")
            self.cache_data = {}
    
    def _cleanup_expired(self):
        """Clean up expired session files and enforce max_sessions limit."""
        if not self.enabled:
            return
        
        try:
            # Get all session files
            session_files = list(self.cache_dir.glob("session_*.pkl"))
            
            # Remove expired sessions
            now = datetime.now()
            valid_sessions = []
            
            for session_file in session_files:
                file_mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                age = now - file_mtime
                
                if age.total_seconds() > self.ttl:
                    # Expired, delete it
                    try:
                        session_file.unlink()
                    except Exception:
                        pass
                else:
                    valid_sessions.append((session_file, file_mtime))
            
            # Enforce max_sessions limit (keep most recent)
            if len(valid_sessions) > self.max_sessions:
                # Sort by modification time (newest first)
                valid_sessions.sort(key=lambda x: x[1], reverse=True)
                
                # Delete oldest sessions beyond max_sessions
                for session_file, _ in valid_sessions[self.max_sessions:]:
                    try:
                        session_file.unlink()
                    except Exception:
                        pass
            
            # Update metadata file
            self._save_metadata()
            
        except Exception as e:
            print(f"[WARN] Failed to cleanup expired sessions: {e}")
    
    def _save_metadata(self):
        """Save session metadata to disk for tracking."""
        if not self.enabled or self.single_file:
            return
        
        try:
            # Get list of all current session files
            session_files = list(self.cache_dir.glob("session_*.pkl"))
            
            metadata = {
                'sessions': [],
                'last_cleanup': datetime.now().isoformat()
            }
            
            for session_file in session_files:
                file_mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                file_size = session_file.stat().st_size
                
                # Parse session info from filename
                parts = session_file.stem.split('_')
                if len(parts) >= 4:
                    team_id = parts[1]
                    gw = parts[2]
                    date = parts[3]
                    
                    metadata['sessions'].append({
                        'filename': session_file.name,
                        'team_id': team_id,
                        'gameweek': gw,
                        'date': date,
                        'created': file_mtime.isoformat(),
                        'size_bytes': file_size
                    })
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"[WARN] Failed to save metadata: {e}")
    
    def get(self, cache_type: str, *args, **kwargs) -> Optional[Any]:
        """Retrieve data from session cache.
        
        Args:
            cache_type: Type of cache
            *args: Arguments that identify the cached data
            **kwargs: Additional arguments
        
        Returns:
            Cached data if found, None otherwise
        """
        if not self.enabled:
            return None
        
        key = self._generate_cache_key(cache_type, *args, **kwargs)
        return self.cache_data.get(key)
    
    def set(self, cache_type: str, data: Any, *args, **kwargs):
        """Store data in session cache.
        
        Args:
            cache_type: Type of cache
            data: Data to cache
            *args: Arguments that identify the cached data
            **kwargs: Additional arguments
        """
        if not self.enabled:
            return
        
        key = self._generate_cache_key(cache_type, *args, **kwargs)
        self.cache_data[key] = data
    
    def save(self):
        """Save session cache to disk.
        
        Call this at the end of a successful report generation to persist
        the session cache for future runs.
        """
        if not self.enabled:
            return
        
        try:
            # Bundle cache data and metadata
            session_bundle = {
                'cache_data': self.cache_data,
                'metadata': self.session_metadata
            }
            
            # Save to disk
            with open(self.session_file, 'wb') as f:
                pickle.dump(session_bundle, f)
            
            # Update metadata file (disabled in single-file mode)
            self._save_metadata()
            
        except Exception as e:
            print(f"[WARN] Failed to save session {self.session_id}: {e}")
    
    def invalidate(self):
        """Invalidate and delete the current session cache."""
        if not self.enabled:
            return
        
        self.cache_data.clear()
        
        if self.session_file.exists():
            try:
                self.session_file.unlink()
            except Exception as e:
                print(f"[WARN] Failed to delete session file: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {'enabled': False}
        
        # Count entries by cache type
        type_counts = {}
        for key in self.cache_data.keys():
            cache_type = key.split('|')[0] if '|' in key else 'unknown'
            type_counts[cache_type] = type_counts.get(cache_type, 0) + 1
        
        # Get session file info
        session_size = 0
        session_exists = False
        if self.session_file.exists():
            session_exists = True
            session_size = self.session_file.stat().st_size
        
        # Count total sessions in cache dir
        if self.single_file:
            all_sessions = [self.session_file] if self.session_file.exists() else []
        else:
            all_sessions = list(self.cache_dir.glob("session_*.pkl"))
        
        return {
            'enabled': True,
            'session_id': self.session_id,
            'team_id': self.team_id,
            'gameweek': self.gameweek,
            'entries_in_memory': len(self.cache_data),
            'entries_by_type': type_counts,
            'session_file_exists': session_exists,
            'session_size_bytes': session_size,
            'session_size_mb': round(session_size / (1024 * 1024), 2),
            'total_sessions_on_disk': len(all_sessions),
            'cache_dir': str(self.cache_dir)
        }
    
    @staticmethod
    def cleanup_all_expired(cache_dir: Optional[Path] = None, ttl: int = 3600):
        """Static method to cleanup all expired sessions.
        
        Args:
            cache_dir: Directory containing cache files.
            ttl: Time-to-live in seconds.
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache"
        
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            return
        
        try:
            session_files = list(cache_dir.glob("session_*.pkl"))
            now = datetime.now()
            deleted = 0
            
            for session_file in session_files:
                file_mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                age = now - file_mtime
                
                if age.total_seconds() > ttl:
                    try:
                        session_file.unlink()
                        deleted += 1
                    except Exception:
                        pass
            
            if deleted > 0:
                print(f"[INFO] Cleaned up {deleted} expired session(s)")
                
        except Exception as e:
            print(f"[WARN] Failed to cleanup expired sessions: {e}")

