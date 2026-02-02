# LLM Client Module

import os
import io
import base64
import json
from pathlib import Path
from typing import List, Dict, Tuple

from PIL import Image as PIL_Image

# Try to import openai
try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AzureOpenAI = None


def load_config() -> Dict:
    """Load configuration file."""
    config_path = Path(__file__).resolve().parents[1] / "downstream" / "config.json"
    if config_path.exists():
        return json.load(open(config_path, "r"))
    return {}


def create_client(client_type: str = "OpenAI"):
    """
    Create OpenAI or Azure client.
    
    Args:
        client_type: "OpenAI" or "Azure"
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("Please install openai: pip install openai")
    
    config = load_config()
    
    if client_type == "Azure":
        return AzureOpenAI(
            api_key=config.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY", ""),
            api_version=config.get("azure_api_version") or os.environ.get("AZURE_OPENAI_API_VERSION", ""),
            azure_endpoint=config.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        )
    
    return OpenAI(
        base_url=config.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=config.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
    )


def encode_image_to_base64(pil_image: PIL_Image.Image) -> str:
    """Encode PIL image to base64."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_llm_with_images(
    client,
    model_name: str,
    images: List[PIL_Image.Image],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
) -> Tuple[str, Dict]:
    """
    Call multimodal LLM.
    
    Args:
        client: OpenAI/Azure client
        model_name: Model name
        images: PIL image list
        system_prompt: System prompt
        user_prompt: User prompt
        max_tokens: Maximum output tokens
    
    Returns:
        (response_text, token_info)
    """
    base64_images = [encode_image_to_base64(img) for img in images]
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    for img in base64_images
                ],
            ],
        },
    ]
    
    # Adjust parameters based on model
    kwargs = {"model": model_name, "messages": messages}
    if model_name.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0.0
    
    response = client.chat.completions.create(**kwargs)
    
    usage = response.usage
    return response.choices[0].message.content, {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
