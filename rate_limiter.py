"""
Rate limiter module for external APIs.
Handles rate limiting for Slack, Linear, and OpenAI APIs.
"""
import time
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    # Requests per time period
    requests_per_period: int
    # Time period in seconds
    period_seconds: int
    # Maximum concurrent requests (if applicable)
    max_concurrent: Optional[int] = None
    # Whether to use token bucket algorithm (vs. sliding window)
    use_token_bucket: bool = False
    # Token bucket refill rate (tokens per second)
    refill_rate: Optional[float] = None
    # Initial token count (for token bucket)
    initial_tokens: Optional[int] = None

class RateLimiter:
    """Base rate limiter class"""
    
    def __init__(self, name: str, config: RateLimitConfig):
        self.name = name
        self.config = config
        self.request_times: List[float] = []
        self.lock = threading.Lock()
        
        # Token bucket variables (if applicable)
        self.tokens = config.initial_tokens if config.use_token_bucket and config.initial_tokens else config.requests_per_period
        self.last_token_refresh = time.time()
        
        logger.info(f"Initialized rate limiter for {name} with {config.requests_per_period} requests per {config.period_seconds}s")
    
    def _clean_old_requests(self) -> None:
        """Remove request timestamps older than the period"""
        current_time = time.time()
        cutoff_time = current_time - self.config.period_seconds
        
        with self.lock:
            self.request_times = [t for t in self.request_times if t >= cutoff_time]
    
    def _refresh_tokens(self) -> None:
        """Refresh tokens for token bucket algorithm"""
        if not self.config.use_token_bucket or not self.config.refill_rate:
            return
            
        current_time = time.time()
        elapsed = current_time - self.last_token_refresh
        new_tokens = elapsed * self.config.refill_rate
        
        with self.lock:
            self.tokens = min(self.tokens + new_tokens, self.config.requests_per_period)
            self.last_token_refresh = current_time
    
    def check_rate_limit(self) -> bool:
        """
        Check if the request is within rate limits.
        Returns True if request can proceed, False if rate limited.
        """
        if self.config.use_token_bucket:
            return self._check_token_bucket()
        else:
            return self._check_sliding_window()
    
    def _check_sliding_window(self) -> bool:
        """Check rate limits using sliding window algorithm"""
        self._clean_old_requests()
        
        with self.lock:
            if len(self.request_times) >= self.config.requests_per_period:
                logger.warning(f"Rate limit exceeded for {self.name}: {len(self.request_times)} requests in {self.config.period_seconds}s")
                return False
            
            self.request_times.append(time.time())
            return True
    
    def _check_token_bucket(self) -> bool:
        """Check rate limits using token bucket algorithm"""
        self._refresh_tokens()
        
        with self.lock:
            if self.tokens < 1:
                logger.warning(f"Rate limit exceeded for {self.name}: no tokens available")
                return False
            
            self.tokens -= 1
            return True
    
    def wait_if_needed(self) -> float:
        """
        Wait if rate limited, return the wait time in seconds.
        Returns 0 if no wait was needed.
        """
        if self.config.use_token_bucket:
            return self._wait_token_bucket()
        else:
            return self._wait_sliding_window()
    
    def _wait_sliding_window(self) -> float:
        """Wait using sliding window algorithm"""
        self._clean_old_requests()
        
        with self.lock:
            if len(self.request_times) < self.config.requests_per_period:
                self.request_times.append(time.time())
                return 0
            
            # Calculate wait time
            oldest_request = min(self.request_times)
            wait_time = oldest_request + self.config.period_seconds - time.time()
            
            if wait_time > 0:
                logger.info(f"Rate limiting {self.name}: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                
                # After waiting, add the new request time and clean old ones
                self.request_times.append(time.time())
                self._clean_old_requests()
                
                return wait_time
            else:
                # If wait time is negative, we can proceed
                self.request_times.append(time.time())
                return 0
    
    def _wait_token_bucket(self) -> float:
        """Wait using token bucket algorithm"""
        self._refresh_tokens()
        
        with self.lock:
            if self.tokens >= 1:
                self.tokens -= 1
                return 0
            
            # Calculate wait time
            if self.config.refill_rate:
                wait_time = (1 - self.tokens) / self.config.refill_rate
                
                if wait_time > 0:
                    logger.info(f"Rate limiting {self.name}: waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    
                    # After waiting, refresh tokens and proceed
                    self._refresh_tokens()
                    self.tokens -= 1
                    
                    return wait_time
            
            return 0

# Create rate limiters for each external API
slack_limiter = RateLimiter("Slack", RateLimitConfig(
    requests_per_period=50,  # 50 requests per minute (Slack's tier 3 limit)
    period_seconds=60
))

linear_limiter = RateLimiter("Linear", RateLimitConfig(
    requests_per_period=180,  # 3 requests per second (Linear's default limit)
    period_seconds=60
))

openai_limiter = RateLimiter("OpenAI", RateLimitConfig(
    requests_per_period=200,  # Typical OpenAI GPT-4 limit per minute for many accounts
    period_seconds=60,
    use_token_bucket=True,
    refill_rate=3.33,  # Refill at 3.33 tokens per second (200/60)
    initial_tokens=200
))

# Global rate limiter for total application requests
global_limiter = RateLimiter("Global", RateLimitConfig(
    requests_per_period=75,  # 75 requests per minute total across all users
    period_seconds=60
)) 