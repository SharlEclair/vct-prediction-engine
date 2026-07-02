import asyncio
import logging
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_client")

BASE_URL = "http://localhost:3000"
SEMAPHORE = asyncio.Semaphore(5)  # Limit concurrent requests to 5

async def _request(client: httpx.AsyncClient, url: str, params: dict = None, retries: int = 3, backoff: float = 1.0) -> dict:
    """Helper to perform requests with retries, backoff, and rate-limiting."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    async with SEMAPHORE:
        for attempt in range(retries):
            try:
                response = await client.get(url, params=params, headers=headers, timeout=15.0)
                
                # Check for rate-limiting
                if response.status_code == 429:
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning(f"Rate limited (429) on {url}. Retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                sleep_time = backoff * (2 ** attempt)
                logger.error(f"Request attempt {attempt+1} failed for {url}: {e}")
                if attempt == retries - 1:
                    raise e
                await asyncio.sleep(sleep_time)
    raise httpx.RequestError("Max retries exceeded without successful response")

async def get_match_details(match_id: str, client: httpx.AsyncClient = None) -> dict:
    """Retrieves match details from local API for the given match ID."""
    url = f"{BASE_URL}/v2/match/details"
    # Pass both 'id' and 'match_id' to satisfy user specs and local API query requirements
    params = {"id": match_id, "match_id": match_id}
    
    is_local_client = False
    if client is None:
        client = httpx.AsyncClient()
        is_local_client = True
        
    try:
        data = await _request(client, url, params=params)
        return data
    finally:
        if is_local_client:
            await client.aclose()

async def get_player_stats(client: httpx.AsyncClient = None) -> dict:
    """Retrieves player statistics from local API."""
    url = f"{BASE_URL}/v2/stats"
    # Pass mandatory region and timespan query parameters for stats endpoint
    params = {"region": "na", "timespan": "all"}
    
    is_local_client = False
    if client is None:
        client = httpx.AsyncClient()
        is_local_client = True
        
    try:
        data = await _request(client, url, params=params)
        return data
    finally:
        if is_local_client:
            await client.aclose()
