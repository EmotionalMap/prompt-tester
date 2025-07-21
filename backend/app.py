import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration - Fixed model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
# MODEL = "gpt-4o-2024-11-20"
MODEL = "gpt-3.5-turbo"  # Default to GPT-3.5 Turbo for compatibility

# Default parameters
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 1024,
}

# In-memory storage for system prompts
system_prompts = {
    "default": {
        "name": "Default Assistant",
        "prompt": "You are a helpful AI assistant. Provide clear, accurate, and concise responses.",
        "description": "General purpose helpful assistant",
        "created_at": "2024-01-01T00:00:00Z"
    },
}

@app.route("/api/completions", methods=["POST"])
def completions():
    if not OPENAI_API_KEY:
        return jsonify(error="OPENAI_API_KEY not set"), 500

    data = request.get_json() or {}
    system_prompt = data.get("systemPrompt", "")
    user_prompt = data.get("userPrompt", "")
    
    if not user_prompt.strip():
        return jsonify(error="User prompt is required"), 400
    
    # Extract parameters with simplified options
    user_options = data.get("options", {})
    
    # Build OpenAI parameters
    openai_params = {
        "temperature": user_options.get("temperature", DEFAULT_PARAMS["temperature"]),
        "max_tokens": user_options.get("max_tokens", DEFAULT_PARAMS["max_tokens"]),
    }
    
    # Optional parameters
    if "seed" in user_options and user_options["seed"] is not None:
        openai_params["seed"] = user_options["seed"]

    # Build OpenAI payload
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    openai_payload = {
        "model": MODEL,
        "messages": messages,
        **{k: v for k, v in openai_params.items() if v is not None}
    }
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        timeout = int(os.getenv("OPENAI_TIMEOUT", "60"))
        resp = requests.post(OPENAI_API_URL, headers=headers, json=openai_payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        
        # Extract response
        text = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        
        return jsonify({
            "text": text,
            "model": MODEL,
            "parameters_used": openai_params,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            "finish_reason": body["choices"][0].get("finish_reason")
        })
        
    except requests.exceptions.Timeout:
        return jsonify(error="Request timed out"), 504
    except requests.exceptions.RequestException as e:
        return jsonify(error=f"OpenAI request failed: {str(e)}"), 500
    except KeyError as e:
        return jsonify(error=f"Unexpected response format: missing {str(e)}"), 500
    except Exception as e:
        return jsonify(error=f"Unexpected error: {str(e)}"), 500

# System Prompts Management
@app.route("/api/system-prompts", methods=["GET"])
def get_system_prompts():
    """Get all system prompts"""
    return jsonify(system_prompts)

@app.route("/api/system-prompts/<prompt_id>", methods=["GET"])
def get_system_prompt(prompt_id):
    """Get a specific system prompt"""
    if prompt_id not in system_prompts:
        return jsonify(error="System prompt not found"), 404
    return jsonify(system_prompts[prompt_id])

@app.route("/api/system-prompts", methods=["POST"])
def create_system_prompt():
    """Create a new system prompt"""
    data = request.get_json() or {}
    
    # Validate required fields
    required_fields = ["id", "name", "prompt"]
    for field in required_fields:
        if not data.get(field, "").strip():
            return jsonify(error=f"Field '{field}' is required"), 400
    
    prompt_id = data["id"].lower().replace(" ", "_").replace("-", "_")
    
    # Check if ID already exists
    if prompt_id in system_prompts:
        return jsonify(error="System prompt ID already exists"), 409
    
    from datetime import datetime
    system_prompts[prompt_id] = {
        "name": data["name"].strip(),
        "prompt": data["prompt"].strip(),
        "description": data.get("description", "").strip(),
        "created_at": datetime.now().isoformat() + "Z"
    }
    
    return jsonify({
        "id": prompt_id,
        **system_prompts[prompt_id]
    }), 201

@app.route("/api/system-prompts/<prompt_id>", methods=["PUT"])
def update_system_prompt(prompt_id):
    """Update an existing system prompt"""
    if prompt_id not in system_prompts:
        return jsonify(error="System prompt not found"), 404
    
    data = request.get_json() or {}
    
    # Update fields if provided
    if "name" in data and data["name"].strip():
        system_prompts[prompt_id]["name"] = data["name"].strip()
    
    if "prompt" in data and data["prompt"].strip():
        system_prompts[prompt_id]["prompt"] = data["prompt"].strip()
    
    if "description" in data:
        system_prompts[prompt_id]["description"] = data.get("description", "").strip()
    
    return jsonify({
        "id": prompt_id,
        **system_prompts[prompt_id]
    })

@app.route("/api/system-prompts/<prompt_id>", methods=["DELETE"])
def delete_system_prompt(prompt_id):
    """Delete a system prompt"""
    if prompt_id not in system_prompts:
        return jsonify(error="System prompt not found"), 404
    
    # Prevent deleting the default prompt
    if prompt_id == "default":
        return jsonify(error="Cannot delete the default system prompt"), 400
    
    deleted_prompt = system_prompts.pop(prompt_id)
    return jsonify({
        "message": f"System prompt '{deleted_prompt['name']}' deleted successfully",
        "deleted": {
            "id": prompt_id,
            **deleted_prompt
        }
    })

@app.route("/api/test-prompts", methods=["POST"])
def test_prompts():
    """Test the same user prompt against multiple system prompts"""
    data = request.get_json() or {}
    user_prompt = data.get("userPrompt", "")
    prompt_ids = data.get("promptIds", [])
    options = data.get("options", {})
    
    if not user_prompt.strip():
        return jsonify(error="User prompt is required"), 400
    
    if not prompt_ids:
        return jsonify(error="At least one system prompt ID is required"), 400
    
    results = {}
    
    for prompt_id in prompt_ids:
        if prompt_id not in system_prompts:
            results[prompt_id] = {"error": f"System prompt '{prompt_id}' not found"}
            continue
        
        try:
            # Make internal request to completions endpoint
            test_payload = {
                "systemPrompt": system_prompts[prompt_id]["prompt"],
                "userPrompt": user_prompt,
                "options": options
            }
            
            # Simulate internal request
            with app.test_client() as client:
                response = client.post("/api/completions", 
                                     json=test_payload,
                                     headers={"Content-Type": "application/json"})
                
                if response.status_code == 200:
                    response_data = response.get_json()
                    results[prompt_id] = {
                        "system_prompt_name": system_prompts[prompt_id]["name"],
                        "system_prompt_text": system_prompts[prompt_id]["prompt"],
                        "text": response_data["text"],
                        "usage": response_data.get("usage", {}),
                        "finish_reason": response_data.get("finish_reason"),
                        "parameters_used": response_data.get("parameters_used", {})
                    }
                else:
                    error_data = response.get_json()
                    results[prompt_id] = {"error": error_data.get("error", "Unknown error")}
                    
        except Exception as e:
            results[prompt_id] = {"error": str(e)}
    
    return jsonify({
        "user_prompt": user_prompt,
        "results": results,
        "test_count": len(prompt_ids),
        "success_count": len([r for r in results.values() if "error" not in r]),
        "model": MODEL
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": MODEL,
        "api_key_configured": bool(OPENAI_API_KEY),
        "system_prompts_count": len(system_prompts)
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    
    if not OPENAI_API_KEY:
        print("⚠️  WARNING: OPENAI_API_KEY not set. Please configure your API key.")
    
    print(f"🚀 GPT-4o Backend starting on port {port}")
    print(f"🤖 Using model: {MODEL}")
    print(f"📝 Loaded {len(system_prompts)} system prompts")
    
    app.run(host="0.0.0.0", port=port, debug=debug)