#!/bin/bash
# Script to sequentially crawl all TokenMetrics sites
# This script runs each site crawl as a separate process to prevent memory issues

echo "===== TokenMetrics Website Crawler ====="
echo "Starting crawl process at $(date)"
echo ""

# Path to Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/sync_data.py"

# Default parameters
FALLBACK="false"
MAX_WAIT=1800
FORCE=""

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --fallback [auto|true|false]   Whether to use fallback scraper (default: false)"
  echo "  --max-wait SECONDS             Maximum wait time for crawl jobs (default: 1800)"
  echo "  --force                        Force update all pages"
  echo "  --help                         Display this help message"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fallback)
      FALLBACK="$2"
      shift 2
      ;;
    --max-wait)
      MAX_WAIT="$2"
      shift 2
      ;;
    --force)
      FORCE="--force"
      shift
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Validate fallback option
if [[ "$FALLBACK" != "auto" && "$FALLBACK" != "true" && "$FALLBACK" != "false" ]]; then
  echo "Error: fallback option must be 'auto', 'true', or 'false'"
  exit 1
fi

# Function to crawl a site
crawl_site() {
  site=$1
  echo "===== Crawling $site site ====="
  echo "Starting at $(date)"
  
  # Use the absolute path to Python in the conda environment
  PYTHON_PATH="/opt/anaconda3/envs/dev_env/bin/python"
  
  # Run the command with the conda Python
  $PYTHON_PATH "$PYTHON_SCRIPT" crawl --site "$site" --fallback "$FALLBACK" --max-wait "$MAX_WAIT" $FORCE
  
  # Check exit status
  if [ $? -eq 0 ]; then
    echo "✅ Successfully crawled $site site"
  else
    echo "❌ Error crawling $site site"
  fi
  
  echo "Finished at $(date)"
  echo ""
}

# Crawl main site
crawl_site "main"

# Wait a moment before next site
echo "Waiting 30 seconds before next site..."
sleep 30

# Crawl research site
crawl_site "research"

# Wait a moment before next site
echo "Waiting 30 seconds before next site..."
sleep 30

# Crawl blog site
crawl_site "blog"

echo "===== All crawl jobs completed ====="
echo "Process finished at $(date)" 