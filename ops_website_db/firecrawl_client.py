import os
import time
import requests
import json
import logging
from typing import Optional, Dict, List, Any, Union
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("firecrawl_client")

class FirecrawlClient:
    """Client for interacting with the Firecrawl API."""
    
    def __init__(self, api_key: str):
        """
        Initialize the Firecrawl client.
        
        Args:
            api_key: Firecrawl API key
        """
        self.api_key = api_key
        self.base_url = "https://api.firecrawl.dev/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.rate_limit_wait = 1.0  # Wait 1 second between API calls to avoid rate limiting
        self.max_retries = 3

    def get_api_key(self):
        """Get the API key used by this client."""
        return self.api_key

    def _make_api_request(self, endpoint: str, data: dict = None, method: str = "POST") -> Dict[str, Any]:
        """
        Make an API request with rate limiting and retry logic.
        
        Args:
            endpoint: API endpoint (relative path)
            data: Request payload
            method: HTTP method
            
        Returns:
            Response data as a dictionary
            
        Raises:
            Exception: If the request fails after all retries
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                # Wait to respect rate limits
                if attempt > 0:
                    wait_time = self.rate_limit_wait * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Rate limit hit. Waiting {wait_time:.2f}s before retry {attempt+1}/{self.max_retries}")
                    time.sleep(wait_time)
                
                response = requests.request(
                    method=method,
                    url=url, 
                    json=data, 
                    headers=self.headers
                )
                
                # Handle rate limiting explicitly
                if response.status_code == 429:
                    logger.warning(f"Rate limit hit. Will retry {attempt+1}/{self.max_retries}")
                    continue
                    
                # For debugging: log full response for non-2xx responses
                if response.status_code >= 400:
                    try:
                        error_body = response.json()
                        logger.error(f"API Error ({response.status_code}): {json.dumps(error_body, indent=2)}")
                    except:
                        try:
                            logger.error(f"API Error ({response.status_code}): {response.text[:500]}")
                        except:
                            logger.error(f"API Error ({response.status_code}): Could not parse response body")
                
                # Raise exception for non-2xx responses
                response.raise_for_status()
                
                # If we get here, the request was successful
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429 and attempt < self.max_retries - 1:
                    # This will be handled in the next iteration
                    continue
                else:
                    logger.error(f"HTTP error: {e}")
                    # Try to extract more error details
                    try:
                        error_response = response.json()
                        if "error" in error_response:
                            logger.error(f"Error details: {error_response['error']}")
                        if "message" in error_response:
                            logger.error(f"Error message: {error_response['message']}")
                    except:
                        pass
                    raise
            except Exception as e:
                logger.error(f"Error making API request: {str(e)}")
                if attempt >= self.max_retries - 1:
                    raise
        
        # If we get here, all retries failed
        raise Exception(f"Failed to make API request after {self.max_retries} attempts")

    def scrape_website(self, url: str, params: dict = None):
        """
        Legacy wrapper method for backward compatibility.
        Use scrape() instead for new code.
        """
        return self.scrape(url=url, **params if params else {})
    
    def crawl_website(self, url: str, params: dict = None):
        """
        Legacy wrapper method for backward compatibility.
        Use crawl() instead for new code.
        """
        return self.crawl(url=url, **params if params else {})
    
    def scrape(self, 
               url: str,
               formats: List[str] = ["markdown"],
               only_main_content: bool = True,
               include_tags: List[str] = None,
               exclude_tags: List[str] = None,
               headers: Dict[str, str] = None,
               wait_for: int = 0,
               mobile: bool = False,
               skip_tls_verification: bool = False,
               timeout: int = 30000,
               json_options: Dict[str, Any] = None,
               actions: List[Dict[str, Any]] = None,
               location: Dict[str, Any] = None,
               remove_base64_images: bool = True,
               block_ads: bool = True,
               proxy: str = "basic",
               change_tracking_options: Dict[str, Any] = None
              ) -> Dict[str, Any]:
        """
        Scrape a single URL using Firecrawl's API.
        
        Args:
            url: Target URL to scrape
            formats: Output formats (e.g., ["markdown", "html", "text"])
            only_main_content: Whether to extract only the main content
            include_tags: HTML tags to include
            exclude_tags: HTML tags to exclude
            headers: Custom HTTP headers for the request
            wait_for: Time to wait for page loading in milliseconds
            mobile: Whether to use mobile user agent
            skip_tls_verification: Whether to skip TLS verification
            timeout: Request timeout in milliseconds
            json_options: Options for JSON extraction
            actions: Browser actions to perform before scraping
            location: Geographic location settings
            remove_base64_images: Whether to remove base64-encoded images
            block_ads: Whether to block advertisements
            proxy: Proxy type to use
            change_tracking_options: Options for tracking changes
            
        Returns:
            Dictionary containing the scraped content and metadata
        """
        logger.info(f"Scraping URL: {url}")
        
        # Build the payload with required and optional parameters
        payload = {
            "url": url,
            "formats": formats,
            "onlyMainContent": only_main_content,
            "waitFor": wait_for,
            "mobile": mobile,
            "skipTlsVerification": skip_tls_verification,
            "timeout": timeout,
            "removeBase64Images": remove_base64_images,
            "blockAds": block_ads,
            "proxy": proxy
        }
        
        # Add optional parameters if provided
        if include_tags:
            payload["includeTags"] = include_tags
        if exclude_tags:
            payload["excludeTags"] = exclude_tags
        if headers:
            payload["headers"] = headers
        if json_options:
            payload["jsonOptions"] = json_options
        if actions:
            payload["actions"] = actions
        if location:
            payload["location"] = location
        if change_tracking_options:
            payload["changeTrackingOptions"] = change_tracking_options
        
        # Make the API request
        try:
            response = self._make_api_request("scrape", payload)
            
            # Handle the new API response format (success/data structure)
            if "success" in response and response.get("success") and "data" in response:
                # New API format
                result = response.get("data", {})
                
                # Create a backwards-compatible structure
                compatible_result = {
                    "content": {},
                    "metadata": result.get("metadata", {})
                }
                
                # Map the content formats
                for format_type in formats:
                    if format_type in result:
                        compatible_result["content"][format_type] = result[format_type]
                
                # Add the original response under _raw key for debugging/transparency
                compatible_result["_raw"] = response
                
                return compatible_result
            
            # Return the original response structure if it doesn't match the new format
            return response
            
        except Exception as e:
            logger.error(f"Failed to scrape URL {url}: {str(e)}")
            return {}
            
    def get_crawl_results(self, crawl_id: str, wait_complete: bool = False, poll_interval: int = 5, max_wait_time: int = 300) -> Dict[str, Any]:
        """
        Get the results of a crawl job by ID.
        
        Args:
            crawl_id: The crawl job ID
            wait_complete: Whether to wait for the crawl to complete
            poll_interval: How many seconds to wait between polling attempts
            max_wait_time: Maximum number of seconds to wait for results
            
        Returns:
            Dictionary containing the crawl results
        """
        logger.info(f"Getting results for crawl job: {crawl_id}")
        
        endpoint = f"crawl/{crawl_id}"
        start_time = time.time()
        
        while True:
            try:
                # Make a GET request to the crawl job endpoint
                response = self._make_api_request(endpoint, method="GET")
                
                # Log the full response for debugging
                try:
                    logger.debug(f"Full crawl result response: {json.dumps(response, indent=2)[:1000]}...")
                except:
                    logger.debug(f"Could not dump response as JSON: {type(response)}")
                
                # Check if the crawl is complete
                status = response.get("status", "").lower()
                
                if status == "completed":
                    logger.info(f"Crawl job {crawl_id} completed successfully")
                    
                    # Log the response structure to debug what fields are available
                    logger.info(f"Response keys: {', '.join(response.keys())}")
                    
                    # NEW FORMAT: The data is an array of pages, not an object with pages
                    if "data" in response and isinstance(response["data"], list):
                        # Create a backwards-compatible structure
                        compatible_result = {
                            "pages": [],
                            "status": "completed"
                        }
                        
                        # Process pages directly from data array
                        pages = response["data"]
                        logger.info(f"Found {len(pages)} pages in data array")
                        
                        for page in pages:
                            # Convert each page to the old format
                            compatible_page = {
                                "url": page.get("metadata", {}).get("url", ""),
                                "content": {
                                    # Move content to expected location
                                    "markdown": page.get("markdown", "")
                                },
                                "metadata": page.get("metadata", {})
                            }
                            
                            compatible_result["pages"].append(compatible_page)
                        
                        # Add the original response under _raw key
                        compatible_result["_raw"] = response
                        
                        return compatible_result
                        
                    # Handle the older API response format with data.pages
                    elif "data" in response and isinstance(response["data"], dict) and "pages" in response["data"]:
                        # Create a backwards-compatible structure
                        compatible_result = {
                            "pages": []
                        }
                        
                        # Process pages array
                        pages = response["data"]["pages"]
                        logger.info(f"Found {len(pages)} pages in the response")
                        
                        for page in pages:
                            # Convert each page to the old format
                            compatible_page = {
                                "url": page.get("url", ""),
                                "content": {},
                                "metadata": page.get("metadata", {})
                            }
                            
                            # Map content formats
                            for format_type in ["markdown", "html", "rawHtml"]:
                                if format_type in page:
                                    compatible_page["content"][format_type] = page[format_type]
                            
                            compatible_result["pages"].append(compatible_page)
                        
                        # Add the original response under _raw key
                        compatible_result["_raw"] = response
                        
                        return compatible_result
                    elif "result" in response and isinstance(response["result"], dict) and "pages" in response["result"]:
                        # Try alternative response structure
                        compatible_result = {
                            "pages": [], 
                            "status": "completed"
                        }
                        
                        # Process pages array
                        pages = response["result"]["pages"]
                        logger.info(f"Found {len(pages)} pages in response['result']['pages']")
                        
                        for page in pages:
                            # Convert each page to the old format
                            compatible_page = {
                                "url": page.get("url", ""),
                                "content": {},
                                "metadata": page.get("metadata", {})
                            }
                            
                            # Map content formats
                            for format_type in ["markdown", "html", "rawHtml"]:
                                if format_type in page:
                                    compatible_page["content"][format_type] = page[format_type]
                            
                            compatible_result["pages"].append(compatible_page)
                        
                        # Add the original response under _raw key
                        compatible_result["_raw"] = response
                        
                        return compatible_result
                    else:
                        # Check other possible locations for pages
                        if "pages" in response:
                            logger.info(f"Found pages at root level: {len(response['pages'])}")
                            return {"pages": response["pages"], "status": status}
                        
                        # Let's see what's actually in the response
                        logger.warning("Could not find 'pages' in the expected location of the response")
                        
                        # Return the original response with status for debugging
                        response["status"] = status
                        return response
                    
                elif status == "failed":
                    error = response.get("error", "Unknown error")
                    logger.error(f"Crawl job {crawl_id} failed: {error}")
                    return {"error": error, "status": "failed"}
                    
                elif status == "processing" or status == "queued" or status == "scraping":
                    if wait_complete:
                        elapsed = time.time() - start_time
                        
                        # Check if we've exceeded the maximum wait time
                        if elapsed > max_wait_time:
                            logger.warning(f"Maximum wait time exceeded for crawl job {crawl_id}")
                            
                            # For scraping status, try to return partial results that are available
                            if status == "scraping" and "data" in response and isinstance(response["data"], list):
                                logger.info(f"Returning partial results with {len(response['data'])} pages from a job in progress")
                                
                                # Create a result with the available pages
                                compatible_result = {
                                    "pages": [],
                                    "status": "partial_results",
                                    "message": "Maximum wait time exceeded, returning partial results"
                                }
                                
                                # Process pages directly from data array
                                pages = response["data"]
                                logger.info(f"Found {len(pages)} pages in data array")
                                
                                for page in pages:
                                    # Convert each page to the old format
                                    compatible_page = {
                                        "url": page.get("metadata", {}).get("url", "") or page.get("metadata", {}).get("sourceURL", ""),
                                        "content": {
                                            # Move content to expected location
                                            "markdown": page.get("markdown", "")
                                        },
                                        "metadata": page.get("metadata", {})
                                    }
                                    
                                    compatible_result["pages"].append(compatible_page)
                                
                                # Add the original response under _raw key
                                compatible_result["_raw"] = response
                                
                                return compatible_result
                            else:
                                return {"status": status, "message": "Maximum wait time exceeded", "job_id": crawl_id}
                            
                        # Progress info
                        total = response.get("total", 0)
                        completed = response.get("completed", 0)
                        
                        if total > 0:
                            percent = round(completed / total * 100)
                            logger.info(f"Crawl job {crawl_id} is {status} ({percent}% complete: {completed}/{total} pages). Polling again in {poll_interval}s...")
                        else:
                            logger.info(f"Crawl job {crawl_id} is {status}. Polling again in {poll_interval}s...")
                            
                        # Wait before polling again
                        time.sleep(poll_interval)
                        continue
                    else:
                        # Don't wait, just return current status
                        return response
                else:
                    logger.warning(f"Unknown status '{status}' for crawl job {crawl_id}")
                    
                    # Even if the status is unknown, try to return partial results if available
                    if "data" in response and isinstance(response["data"], list) and len(response["data"]) > 0:
                        logger.info(f"Found {len(response['data'])} pages despite unknown status, returning them")
                        
                        # Create a result with the available pages
                        compatible_result = {
                            "pages": [],
                            "status": status
                        }
                        
                        # Process pages directly from data array
                        pages = response["data"]
                        
                        for page in pages:
                            # Convert each page to the old format
                            compatible_page = {
                                "url": page.get("metadata", {}).get("url", "") or page.get("metadata", {}).get("sourceURL", ""),
                                "content": {
                                    # Move content to expected location
                                    "markdown": page.get("markdown", "")
                                },
                                "metadata": page.get("metadata", {})
                            }
                            
                            compatible_result["pages"].append(compatible_page)
                        
                        # Add the original response under _raw key
                        compatible_result["_raw"] = response
                        
                        return compatible_result
                    
                    return response
                    
            except Exception as e:
                logger.error(f"Error getting crawl results for job {crawl_id}: {str(e)}")
                traceback.print_exc()
                return {"error": str(e), "status": "error", "job_id": crawl_id}
                
    def crawl(self,
              url: str,
              exclude_paths: List[str] = None,
              include_paths: List[str] = None,
              max_depth: int = 10,
              max_discovery_depth: int = None,
              ignore_sitemap: bool = False,
              ignore_query_parameters: bool = False,
              limit: int = 100,
              allow_backward_links: bool = False,
              allow_external_links: bool = False,
              webhook: Dict[str, Any] = None,
              scrape_options: Dict[str, Any] = None,
              wait_for_results: bool = False,
              poll_interval: int = 5,
              max_wait_time: int = 300
             ) -> Dict[str, Any]:
        """
        Crawl a website starting from a URL using Firecrawl's API.
        
        Args:
            url: Starting URL for crawling
            exclude_paths: URL paths to exclude from crawling
            include_paths: URL paths to include in crawling
            max_depth: Maximum crawl depth
            max_discovery_depth: Maximum discovery depth
            ignore_sitemap: Whether to ignore the site's sitemap
            ignore_query_parameters: Whether to ignore query parameters in URLs
            limit: Maximum number of pages to crawl
            allow_backward_links: Whether to follow links pointing to previously visited paths
            allow_external_links: Whether to follow links to external domains
            webhook: Webhook configuration for notifications
            scrape_options: Options for scraping individual pages
            wait_for_results: Whether to wait for crawl to complete (can be slow)
            poll_interval: Seconds to wait between polling attempts
            max_wait_time: Maximum seconds to wait for crawl to complete
            
        Returns:
            Dictionary containing the crawl results and metadata
        """
        logger.info(f"Crawling URL: {url}")
        
        # Build the payload with required and optional parameters
        payload = {
            "url": url,
            "maxDepth": max_depth,
            "ignoreSitemap": ignore_sitemap,
            "ignoreQueryParameters": ignore_query_parameters,
            "limit": limit,
            "allowBackwardLinks": allow_backward_links,
            "allowExternalLinks": allow_external_links
        }
        
        # Add optional parameters if provided
        if exclude_paths:
            payload["excludePaths"] = exclude_paths
        if include_paths:
            payload["includePaths"] = include_paths
        if max_discovery_depth:
            payload["maxDiscoveryDepth"] = max_discovery_depth
        if webhook:
            payload["webhook"] = webhook
        if scrape_options:
            payload["scrapeOptions"] = scrape_options
        else:
            # Default scrape options if none provided
            payload["scrapeOptions"] = {
                "formats": ["markdown"],
                "onlyMainContent": True,
                "removeBase64Images": True,
                "blockAds": True
            }
        
        # Make the API request
        try:
            response = self._make_api_request("crawl", payload)
            
            # Handle asynchronous crawl - job ID is returned instead of results
            if "success" in response and response.get("success") and "id" in response:
                job_id = response.get("id")
                logger.info(f"Crawl job started with ID: {job_id}")
                
                if wait_for_results:
                    logger.info(f"Waiting for crawl job to complete (up to {max_wait_time} seconds)...")
                    return self.get_crawl_results(job_id, wait_complete=True, 
                                                poll_interval=poll_interval,
                                                max_wait_time=max_wait_time)
                else:
                    return response
            
            # Handle the older synchronous API response format
            elif "success" in response and response.get("success") and "data" in response:
                # Create a backwards-compatible structure 
                return self._process_crawl_response(response)
            
            # Return the original response if it doesn't match any known format
            return response
            
        except Exception as e:
            logger.error(f"Failed to crawl URL {url}: {str(e)}")
            return {"pages": []}
    
    def _process_crawl_response(self, response):
        """Process a crawl response into a backwards-compatible format"""
        # New API format with success/data structure
        data = response.get("data", {})
        
        # Create a backwards-compatible structure
        compatible_result = {
            "pages": []
        }
        
        # Process pages array if it exists
        if "pages" in data and isinstance(data["pages"], list):
            for page in data["pages"]:
                # Convert each page to the old format
                compatible_page = {
                    "url": page.get("url", ""),
                    "content": {},
                    "metadata": page.get("metadata", {})
                }
                
                # Map content formats
                for format_type in ["markdown", "html", "rawHtml"]:
                    if format_type in page:
                        compatible_page["content"][format_type] = page[format_type]
                
                compatible_result["pages"].append(compatible_page)
        
        # Add the original response under _raw key
        compatible_result["_raw"] = response
        
        return compatible_result
