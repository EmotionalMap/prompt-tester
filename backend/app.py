import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() == "true"
MODEL = os.getenv("MODEL", "deepseek-r1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# OpenAI-like defaults for consistent behavior
OPENAI_LIKE_DEFAULTS = {
    "temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", "1024")),
    "seed": None
}

# Ollama parameters configured to match OpenAI behavior
OLLAMA_OPENAI_LIKE = {
    "top_k": 50,           # Higher diversity like OpenAI
    "top_p": 1.0,          # No nucleus sampling restriction
    "repeat_penalty": 1.0,  # Disable repeat penalty (OpenAI doesn't use this)
    "num_ctx": 4096        # Larger context window
}

@app.route("/api/completions", methods=["POST"])
def completions():
    data = request.get_json() or {}
    system_prompt = data.get("systemPrompt", "")
    user_prompt = data.get("userPrompt", "")
    
    # Extract only migration-relevant parameters
    user_options = data.get("options", {})
    
    # Build parameters that work for both Ollama and OpenAI
    migration_params = {}
    if "temperature" in user_options:
        migration_params["temperature"] = user_options["temperature"]
    else:
        migration_params["temperature"] = OPENAI_LIKE_DEFAULTS["temperature"]
        
    if "max_tokens" in user_options:
        migration_params["max_tokens"] = user_options["max_tokens"]
    else:
        migration_params["max_tokens"] = OPENAI_LIKE_DEFAULTS["max_tokens"]
        
    if "seed" in user_options:
        migration_params["seed"] = user_options["seed"]

    if USE_LOCAL:
        # Build Ollama payload
        ollama_options = {
            "temperature": migration_params["temperature"],
            "num_predict": migration_params["max_tokens"],  # Ollama uses num_predict instead of max_tokens
            **OLLAMA_OPENAI_LIKE  # Add OpenAI-like Ollama-specific params
        }
        
        if migration_params.get("seed"):
            ollama_options["seed"] = migration_params["seed"]
        
        ollama_payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": ollama_options
        }
        
        try:
            timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
            resp = requests.post(OLLAMA_URL, json=ollama_payload, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            
            text = body["message"]["content"]
            
            return jsonify({
                "text": text,
                "model": MODEL,
                "backend": "ollama",
                "migration_params": migration_params,  # Show what will transfer to OpenAI
                "performance": {
                    "total_duration": body.get("total_duration"),
                    "eval_count": body.get("eval_count"),
                    "tokens_per_second": body.get("eval_count", 0) / (body.get("eval_duration", 1) / 1e9) if body.get("eval_duration") else 0
                }
            })
            
        except Exception as e:
            return jsonify(error=f"Ollama request failed: {str(e)}"), 500

    else:
        # OpenAI branch - uses migration_params directly
        if not OPENAI_API_KEY:
            return jsonify(error="OPENAI_API_KEY not set"), 500

        openai_payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **{k: v for k, v in migration_params.items() if v is not None}  # Only include non-None values
        }
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            resp = requests.post(OPENAI_API_URL, headers=headers, json=openai_payload)
            resp.raise_for_status()
            body = resp.json()
            
            text = body["choices"][0]["message"]["content"]
            
            return jsonify({
                "text": text,
                "model": openai_payload["model"],
                "backend": "openai",
                "migration_params": migration_params,
                "performance": body.get("usage", {})
            })
            
        except Exception as e:
            return jsonify(error=f"OpenAI request failed: {str(e)}"), 500

@app.route("/api/presets", methods=["GET"])
def presets():
    """Return presets focused on parameters that migrate well"""
    return jsonify({
        "creative": {
            "temperature": 0.9,
            "max_tokens": 400,
            "description": "High creativity - good for brainstorming, stories"
        },
        "balanced": {
            "temperature": 0.7,
            "max_tokens": 300,
            "description": "Balanced - good for general conversation"
        },
        "focused": {
            "temperature": 0.3,
            "max_tokens": 200,
            "description": "Low temperature - good for factual, analytical tasks"
        },
        "precise": {
            "temperature": 0.1,
            "max_tokens": 150,
            "description": "Very focused - good for technical accuracy"
        }
    })

@app.route("/api/switch-backend", methods=["POST"])
def switch_backend():
    """Helper endpoint to test the same prompt on both backends"""
    data = request.get_json() or {}
    system_prompt = data.get("systemPrompt", "")
    user_prompt = data.get("userPrompt", "")
    user_options = data.get("options", {})
    
    results = {}
    
    # Test with Ollama
    try:
        # Make direct request to test Ollama
        migration_params = {
            "temperature": user_options.get("temperature", OPENAI_LIKE_DEFAULTS["temperature"]),
            "max_tokens": user_options.get("max_tokens", OPENAI_LIKE_DEFAULTS["max_tokens"])
        }
        
        ollama_options = {
            "temperature": migration_params["temperature"],
            "num_predict": migration_params["max_tokens"],
            **OLLAMA_OPENAI_LIKE
        }
        
        if migration_params.get("seed"):
            ollama_options["seed"] = migration_params["seed"]
        
        ollama_payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": ollama_options
        }
        
        resp = requests.post(OLLAMA_URL, json=ollama_payload, timeout=30)
        if resp.status_code == 200:
            body = resp.json()
            results["ollama"] = {
                "text": body["message"]["content"],
                "model": MODEL,
                "backend": "ollama"
            }
        else:
            results["ollama"] = {"error": f"Ollama returned status {resp.status_code}"}
    except Exception as e:
        results["ollama"] = {"error": f"Ollama request failed: {str(e)}"}
    
    # Test with OpenAI (if key available)
    if OPENAI_API_KEY:
        try:
            migration_params = {
                "temperature": user_options.get("temperature", OPENAI_LIKE_DEFAULTS["temperature"]),
                "max_tokens": user_options.get("max_tokens", OPENAI_LIKE_DEFAULTS["max_tokens"])
            }
            
            openai_payload = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **{k: v for k, v in migration_params.items() if v is not None}
            }
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            resp = requests.post(OPENAI_API_URL, headers=headers, json=openai_payload, timeout=30)
            if resp.status_code == 200:
                body = resp.json()
                results["openai"] = {
                    "text": body["choices"][0]["message"]["content"],
                    "model": openai_payload["model"],
                    "backend": "openai"
                }
            else:
                results["openai"] = {"error": f"OpenAI returned status {resp.status_code}"}
        except Exception as e:
            results["openai"] = {"error": f"OpenAI request failed: {str(e)}"}
    else:
        results["openai"] = {"error": "OPENAI_API_KEY not set"}
    
    return jsonify({
        **results,
        "migration_ready": bool(results.get("ollama") and not results["ollama"].get("error"))
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)