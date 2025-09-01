# 🔧 Ollama Fix Guide

## **Issue: 500 Server Error with Models**

Your models are returning 500 errors, which usually means they're corrupted or have memory issues.

## **Quick Fix Steps:**

### **Step 1: Stop Ollama**
```bash
# Windows
taskkill /f /im ollama.exe

# Linux/Mac
pkill ollama
```

### **Step 2: Remove Corrupted Models**
```bash
ollama rm mistral:latest
ollama rm llama2:latest
```

### **Step 3: Pull Smaller Model**
```bash
ollama pull llama2:7b
```

### **Step 4: Test the Model**
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2:7b",
    "prompt": "Hello",
    "options": {"num_gpu": 0}
  }'
```

### **Step 5: Update Your Environment**
Create a `.env` file in your project root:
```env
LLM_PROVIDER=ollama
LLM_MODEL_NAME=llama2:7b
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000
```

## **Alternative Solutions:**

### **Option 1: Use CPU Only**
```bash
# Set environment variable
set OLLAMA_NUM_GPU=0

# Start Ollama
ollama serve
```

### **Option 2: Pull Even Smaller Model**
```bash
ollama pull llama2:3b
```

### **Option 3: Use Different Model**
```bash
ollama pull codellama:7b
```

## **Why This Happens:**

1. **GPU Memory Issues**: Your GPU doesn't have enough memory for the large models
2. **Corrupted Downloads**: Models may have been corrupted during download
3. **Version Conflicts**: Model versions may be incompatible

## **Prevention:**

1. **Use CPU Mode**: Set `OLLAMA_NUM_GPU=0` for CPU-only operation
2. **Smaller Models**: Use 7B or 3B models instead of larger ones
3. **Regular Updates**: Keep Ollama updated

## **Test Your Fix:**

Run this in your application:
```python
python fix_ollama.py
```

Or test manually:
```bash
ollama run llama2:7b "Hello, this is a test"
```

## **If Still Not Working:**

1. **Reinstall Ollama**: Download fresh from https://ollama.ai
2. **Clear Cache**: Delete `~/.ollama` folder
3. **Use Different Port**: Try `OLLAMA_HOST=127.0.0.1:11435`

## **Success Indicators:**

✅ Models list shows `llama2:7b`  
✅ Test request returns 200 status  
✅ No CUDA memory errors  
✅ Coaching system works with AI feedback  

---

**Need Help?** Check the logs in your terminal for specific error messages.
