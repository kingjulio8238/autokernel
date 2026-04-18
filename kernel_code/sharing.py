"""Share optimization results via HF Hub link."""
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

def share_session(session_id: str, session_data: dict,
                  hub_token: str | None = None) -> str | None:
    """Upload session results to HF Hub and return a shareable URL.
    Returns the URL or None if upload fails."""
    try:
        from openkernel.hub.client import HubClient
        from openkernel.config import HubConfig
        config = HubConfig(token=hub_token)
        client = HubClient(config)

        # Save session data as JSON
        temp_path = Path(f".kernel-code/shared/{session_id}.json")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(session_data, indent=2, default=str))

        # Upload to HF Hub
        client.upload_file(
            repo_id=config.results_repo,
            local_path=str(temp_path),
            path_in_repo=f"shared/{session_id}.json",
        )

        url = f"https://huggingface.co/datasets/{config.results_repo}/blob/main/shared/{session_id}.json"
        return url
    except Exception as e:
        logger.warning(f"Failed to share session: {e}")
        return None

def format_share_result(url: str | None, session_id: str) -> str:
    """Format the sharing result for display."""
    if url:
        return f"Session shared: {url}"
    return "Sharing failed. Check HF_TOKEN and try again."
