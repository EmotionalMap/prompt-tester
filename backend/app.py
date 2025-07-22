import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4o-2024-11-20"

DEFAULT_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 1024,
}

# File-based storage
PROMPTS_FILE = "system_prompts.json"

def load_prompts_from_file():
    """Load system prompts from JSON file"""
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading prompts file: {e}")
            return get_default_prompts()
    else:
        print(f"Creating new prompts file: {PROMPTS_FILE}")
        return get_default_prompts()

def save_prompts_to_file(prompts):
    """Save system prompts to JSON file"""
    try:
        with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(prompts)} prompts to {PROMPTS_FILE}")
    except IOError as e:
        print(f"Error saving prompts file: {e}")

def get_default_prompts():
    """Return default system prompts"""
    return {
        "default": {
            "name": "Default Assistant",
            "modules": {
                "DEFAULT": "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
            },
            "order": ["DEFAULT"],
            "description": "General purpose helpful assistant",
            "created_at": "2024-01-01T00:00:00Z"
        }
    }

# Load prompts on startup
system_prompts = load_prompts_from_file()

@app.route("/api/completions", methods=["POST"])
def completions():
    if not OPENAI_API_KEY:
        return jsonify(error="OPENAI_API_KEY not set"), 500

    data = request.get_json() or {}
    # Backwards-compatible: allow passing a full systemPrompt string
    system_prompt_text = data.get("systemPrompt", "").strip()
    user_prompt = data.get("userPrompt", "").strip()
    prompt_id = data.get("promptId")
    conversation_history = data.get("conversationHistory", [])

    if not user_prompt:
        return jsonify(error="User prompt is required"), 400

    # If promptId provided, assemble modules into one string
    if prompt_id:
        prompt_obj = system_prompts.get(prompt_id)
        if not prompt_obj:
            return jsonify(error=f"System prompt '{prompt_id}' not found"), 404
        modules = prompt_obj.get("modules", {})
        order = prompt_obj.get("order", [])
        # join in order, skipping missing modules
        system_prompt_text = "\n\n".join(modules.get(name, "") for name in order)

    # Build OpenAI parameters
    user_opts = data.get("options", {})
    openai_params = {
        "temperature": user_opts.get("temperature", DEFAULT_PARAMS["temperature"]),
        "max_tokens": user_opts.get("max_tokens", DEFAULT_PARAMS["max_tokens"]),
    }
    if "seed" in user_opts:
        openai_params["seed"] = user_opts["seed"]

    # Build messages with conversation history
    messages = []
    if system_prompt_text:
        messages.append({"role": "system", "content": system_prompt_text})
    
    # Add conversation history
    for msg in conversation_history:
        if msg.get("role") in ["user", "assistant"] and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current user message
    messages.append({"role": "user", "content": user_prompt})

    payload = {
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
        resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]["message"]
        usage = body.get("usage", {})

        return jsonify({
            "text": choice["content"],
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

# --- System Prompts CRUD ---

@app.route("/api/system-prompts", methods=["GET"])
def get_system_prompts():
    return jsonify(system_prompts)

@app.route("/api/system-prompts/<prompt_id>", methods=["GET"])
def get_system_prompt(prompt_id):
    prompt = system_prompts.get(prompt_id)
    if not prompt:
        return jsonify(error="System prompt not found"), 404
    return jsonify(prompt)

@app.route("/api/system-prompts", methods=["POST"])
def create_system_prompt():
    global system_prompts
    data = request.get_json() or {}
    required = ["id", "name", "modules", "order"]
    for field in required:
        if not data.get(field):
            return jsonify(error=f"Field '{field}' is required"), 400

    pid = data["id"].lower().replace(" ", "_").replace("-", "_")
    if pid in system_prompts:
        return jsonify(error="System prompt ID already exists"), 409

    system_prompts[pid] = {
        "name": data["name"].strip(),
        "modules": data["modules"],
        "order": data["order"],
        "description": data.get("description", "").strip(),
        "created_at": datetime.now().isoformat() + "Z"
    }
    
    # Save to file
    save_prompts_to_file(system_prompts)
    
    return jsonify({"id": pid, **system_prompts[pid]}), 201

@app.route("/api/system-prompts/<prompt_id>", methods=["PUT"])
def update_system_prompt(prompt_id):
    global system_prompts
    if prompt_id not in system_prompts:
        return jsonify(error="System prompt not found"), 404
    data = request.get_json() or {}

    if "name" in data and data["name"].strip():
        system_prompts[prompt_id]["name"] = data["name"].strip()
    if "description" in data:
        system_prompts[prompt_id]["description"] = data["description"].strip()
    if "modules" in data and isinstance(data["modules"], dict):
        # Replace entire modules dict
        system_prompts[prompt_id]["modules"] = data["modules"]
    if "order" in data and isinstance(data["order"], list):
        system_prompts[prompt_id]["order"] = data["order"]
    
    # Save to file
    save_prompts_to_file(system_prompts)

    return jsonify({"id": prompt_id, **system_prompts[prompt_id]})

@app.route("/api/system-prompts/<prompt_id>", methods=["DELETE"])
def delete_system_prompt(prompt_id):
    global system_prompts
    if prompt_id not in system_prompts:
        return jsonify(error="System prompt not found"), 404
    if prompt_id == "default":
        return jsonify(error="Cannot delete the default system prompt"), 400

    deleted = system_prompts.pop(prompt_id)
    
    # Save to file
    save_prompts_to_file(system_prompts)
    
    return jsonify({
        "message": f"System prompt '{deleted['name']}' deleted successfully",
        "deleted": {"id": prompt_id, **deleted}
    })

# --- Test Prompts (compares multiple prompts) ---

@app.route("/api/test-prompts", methods=["POST"])
def test_prompts():
    data = request.get_json() or {}
    user_prompt = data.get("userPrompt", "").strip()
    prompt_ids = data.get("promptIds", [])
    options = data.get("options", {})
    modules_order = data.get("modulesOrder")  # Optional custom module ordering

    if not user_prompt:
        return jsonify(error="User prompt is required"), 400
    if not prompt_ids:
        return jsonify(error="At least one system prompt ID is required"), 400

    results = {}
    for pid in prompt_ids:
        if pid not in system_prompts:
            results[pid] = {"error": f"System prompt '{pid}' not found"}
            continue

        # assemble full prompt
        prompt_obj = system_prompts[pid]
        modules = prompt_obj["modules"]
        
        # Use custom module order if provided, otherwise use saved order
        if modules_order:
            order = modules_order
        else:
            order = prompt_obj.get("order", [])
        
        # Join modules in specified order, skipping missing modules
        full_system = "\n\n".join(modules.get(name, "") for name in order if modules.get(name))

        test_payload = {
            "systemPrompt": full_system,
            "userPrompt": user_prompt,
            "options": options
        }

        with app.test_client() as client:
            resp = client.post("/api/completions", json=test_payload)
            if resp.status_code == 200:
                data_resp = resp.get_json()
                results[pid] = {
                    "system_prompt_name": prompt_obj["name"],
                    "system_prompt_text": full_system,
                    "text": data_resp["text"],
                    "usage": data_resp.get("usage", {}),
                    "finish_reason": data_resp.get("finish_reason"),
                    "parameters_used": data_resp.get("parameters_used", {}),
                    "modules_used": order  # Include which modules were used
                }
            else:
                err = resp.get_json()
                results[pid] = {"error": err.get("error", "Unknown error")}

    return jsonify({
        "user_prompt": user_prompt,
        "results": results,
        "test_count": len(prompt_ids),
        "success_count": len([r for r in results.values() if "error" not in r]),
        "model": MODEL
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": MODEL,
        "api_key_configured": bool(OPENAI_API_KEY),
        "system_prompts_count": len(system_prompts),
        "prompts_file": PROMPTS_FILE,
        "prompts_file_exists": os.path.exists(PROMPTS_FILE)
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set.")
    print(f"Starting on port {port}, model={MODEL}")
    print(f"Prompts file: {PROMPTS_FILE}")
    print(f"Loaded {len(system_prompts)} system prompts")
    app.run(host="0.0.0.0", port=port, debug=debug)