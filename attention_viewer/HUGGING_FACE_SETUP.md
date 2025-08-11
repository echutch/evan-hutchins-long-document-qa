# Hugging Face Authentication Setup

## Step 1: Update your API key
Edit the `.env` file in this directory and replace `YOUR_HUGGING_FACE_API_KEY_HERE` with your actual Hugging Face API key:

```
HUGGING_FACE_HUB_TOKEN=hf_your_actual_api_key_here
```

## Step 2: Ensure you have access to the gated model
For the Llama model (`meta-llama/Llama-3.1-8B-Instruct`), you need to:

1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Request access to this model" 
3. Accept the terms and conditions
4. Wait for approval (usually takes a few minutes to hours)

## Step 3: Test the setup
Once you've updated the `.env` file with your API key and have been granted access to the model, try running:

```bash
python main.py
```

The script will now automatically load your API key from the `.env` file and authenticate with Hugging Face.

## Important Notes
- The `.env` file is already added to `.gitignore`, so your API key won't be committed to version control
- Keep your API key secure and never share it publicly
- If you get a 403 error instead of 401, it means your API key is working but you don't have access to the specific model yet
