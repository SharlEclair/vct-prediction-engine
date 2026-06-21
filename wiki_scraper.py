import httpx
import pandas as pd
import re
import os
import logging
from selectolax.parser import HTMLParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wiki_scraper")

CACHE_DIR = os.path.join(".", "data", "cache")

async def scrape_patch_notes(client: httpx.AsyncClient = None) -> pd.DataFrame:
    """
    Scrapes patch version numbers and release dates from the Valorant Wiki.
    Falls back to a locally cached HTML file if the network request fails or is blocked.
    """
    url = "https://wiki.playvalorant.com/en-us/Patch_Notes"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    is_local_client = False
    if client is None:
        client = httpx.AsyncClient()
        is_local_client = True
        
    html_content = None
    try:
        logger.info(f"Fetching patch notes from {url}...")
        response = await client.get(url, headers=headers, timeout=20.0)
        
        # If response is blocked, trigger Exception
        if response.status_code == 403:
            raise httpx.HTTPStatusError("403 Forbidden (likely Cloudflare block)", request=response.request, response=response)
            
        response.raise_for_status()
        html_content = response.text
        logger.info("Successfully fetched patch notes from live wiki.")
    except Exception as e:
        logger.warning(f"Failed to fetch patch notes from live URL: {e}. Attempting local cache fallback...")
        cache_path = os.path.join(CACHE_DIR, "patch_notes_raw.html")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            logger.info("Successfully loaded patch notes from local cache.")
        else:
            logger.error("Local cache file for patch notes not found.")
            raise e
    finally:
        if is_local_client:
            await client.aclose()

    # Parse the HTML content
    parser = HTMLParser(html_content)
    patches = []
    
    for row in parser.css('tr'):
        tds = row.css('td')
        for i, td in enumerate(tds):
            a = td.css_first('a')
            if a:
                href = a.attributes.get('href', '')
                if '/Patch_Notes/' in href:
                    version = a.text().strip()
                    # Verify version layout (starts with 'v' or digit)
                    if version and (version.lower().startswith('v') or version[0].isdigit()):
                        # Next sibling cell contains the date
                        if i + 1 < len(tds):
                            date_raw = tds[i+1].text()
                            # Clean the date string
                            date_clean = re.sub(r'\[\d+\]', '', date_raw)  # Remove citation brackets e.g. [1]
                            date_clean = re.sub(r'\*', '', date_clean)      # Remove asterisks e.g. *
                            date_clean = date_clean.strip()
                            
                            if date_clean:
                                patches.append({
                                    "patch_version": version,
                                    "release_date": date_clean
                                })
                                
    if not patches:
        logger.warning("No patches extracted from HTML content.")
        return pd.DataFrame(columns=["patch_version", "release_date"])
        
    df = pd.DataFrame(patches)
    # Remove duplicates and clean index
    df = df.drop_duplicates(subset=["patch_version"]).reset_index(drop=True)
    logger.info(f"Successfully processed {len(df)} patch versions.")
    return df

async def scrape_agent_roles(client: httpx.AsyncClient = None) -> dict:
    """
    Scrapes agent character names and tactical roles from the Valorant Wiki.
    Falls back to a locally cached HTML file if the network request fails or is blocked.
    """
    url = "https://wiki.playvalorant.com/en-us/Agents"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    is_local_client = False
    if client is None:
        client = httpx.AsyncClient()
        is_local_client = True
        
    html_content = None
    try:
        logger.info(f"Fetching agent roles from {url}...")
        response = await client.get(url, headers=headers, timeout=20.0)
        
        # If response is blocked, trigger Exception
        if response.status_code == 403:
            raise httpx.HTTPStatusError("403 Forbidden (likely Cloudflare block)", request=response.request, response=response)
            
        response.raise_for_status()
        html_content = response.text
        logger.info("Successfully fetched agent roles from live wiki.")
    except Exception as e:
        logger.warning(f"Failed to fetch agent roles from live URL: {e}. Attempting local cache fallback...")
        cache_path = os.path.join(CACHE_DIR, "agents_raw.html")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            logger.info("Successfully loaded agent roles from local cache.")
        else:
            logger.error("Local cache file for agent roles not found.")
            raise e
    finally:
        if is_local_client:
            await client.aclose()

    # Parse the HTML content
    parser = HTMLParser(html_content)
    agent_roles = {}
    
    for row in parser.css('tr'):
        tds = row.css('td')
        if len(tds) >= 3:
            # Agent name td
            agent_links = tds[1].css('a')
            if not agent_links:
                continue
            agent_name = agent_links[-1].text().strip()
            
            # Tactical role td
            role_raw = tds[2].text()
            role = None
            for r in ["Duelist", "Initiator", "Controller", "Sentinel"]:
                if r.lower() in role_raw.lower():
                    role = r
                    break
            
            if agent_name and role:
                # Filter out helper table or formatting matches
                if agent_name in ["Duelist", "Initiator", "Controller", "Sentinel", "Role"]:
                    continue
                agent_roles[agent_name] = role
                
    logger.info(f"Successfully processed {len(agent_roles)} agent roles.")
    return agent_roles
