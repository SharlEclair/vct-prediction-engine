import os
import logging
import random
import asyncio
import pandas as pd
import re
from curl_cffi import requests
from selectolax.parser import HTMLParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wiki_scraper")

CACHE_DIR = os.path.join(".", "data", "cache")

async def scrape_patch_notes(client=None) -> pd.DataFrame:
    """
    Scrapes patch version numbers and release dates from the Valorant Wiki.
    Falls back to a locally cached HTML file if the network request fails or is blocked.
    """
    url = "https://wiki.playvalorant.com/en-us/Patch_Notes"
    
    is_local_client = False
    if client is None:
        client = requests.AsyncSession(impersonate="chrome")
        is_local_client = True
        
    html_content = None
    try:
        sleep_time = 3.0 + random.uniform(0.5, 2.5)
        logger.info(f"Sleeping for {sleep_time:.2f}s before fetching patch notes...")
        await asyncio.sleep(sleep_time)
        logger.info(f"Fetching patch notes from {url}...")
        response = await client.get(url, timeout=20.0)
        
        if response.status_code != 200:
            raise Exception(f"HTTP Status {response.status_code}")
            
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
            await client.close()

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

async def scrape_agent_roles(client=None) -> dict:
    """
    Scrapes agent character names and tactical roles from the Valorant Wiki.
    Falls back to a locally cached HTML file if the network request fails or is blocked.
    """
    url = "https://wiki.playvalorant.com/en-us/Agents"
    
    is_local_client = False
    if client is None:
        client = requests.AsyncSession(impersonate="chrome")
        is_local_client = True
        
    html_content = None
    try:
        sleep_time = 3.0 + random.uniform(0.5, 2.5)
        logger.info(f"Sleeping for {sleep_time:.2f}s before fetching agent roles...")
        await asyncio.sleep(sleep_time)
        logger.info(f"Fetching agent roles from {url}...")
        response = await client.get(url, timeout=20.0)
        
        if response.status_code != 200:
            raise Exception(f"HTTP Status {response.status_code}")
            
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
        if is_local_client and hasattr(client, "close"):
            await client.close()

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

if __name__ == "__main__":
    import asyncio
    async def main():
        df = await scrape_patch_notes()
        if not df.empty:
            out_dir = os.path.join(".", "data", "raw")
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "patch_notes.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} scraped patch versions to {csv_path}")
            
    asyncio.run(main())
