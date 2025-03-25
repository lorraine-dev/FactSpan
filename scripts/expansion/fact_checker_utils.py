from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Specific URLs that should include the full path
full_path_urls = [
    "youtube.com/channel/UCRAbwEqGDnUBt_gPOkplGBA",
    "facebook.com/bdfactcheck",
    "facebook.com/matsda2sh"
]

def extract_domain_or_path(url):
    """Extracts the domain or full path from a URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path.strip('/')

    if domain.startswith('www.'):
        domain = domain[4:]

    full_path_identifier = f"{domain}/{path}"
    return full_path_identifier if full_path_identifier in full_path_urls else domain

def initialize_valid_fact_checkers():
    """Initializes the set of valid fact-checker domains."""
    try:
        with open('../../Data/ValidFactCheckers/ifcn_list.txt', 'r') as f:
            ifcn_urls = f.read().splitlines()
    except FileNotFoundError:
        logging.error("ifcn_list.txt not found.")
        ifcn_urls = []

    try:
        with open('../../Data/ValidFactCheckers/duke_list.txt', 'r') as f:
            duke_urls = f.read().splitlines()
    except FileNotFoundError:
        logging.error("duke_list.txt not found.")
        duke_urls = []

    valid_domains = set()
    for url in ifcn_urls:
        valid_domains.add(extract_domain_or_path(f'https://{url}'))

    for url in duke_urls:
        valid_domains.add(extract_domain_or_path(url))
    logging.info(f"valid_domains: {len(valid_domains)}")
    return valid_domains

valid_domains = initialize_valid_fact_checkers()

def is_valid_fact_checker(url):
    """Checks if a URL belongs to a valid fact-checker."""
    try:
        return extract_domain_or_path(url) in valid_domains
    except Exception as e:
        logging.error(f"Error checking URL, {url}: {e}")
        return False