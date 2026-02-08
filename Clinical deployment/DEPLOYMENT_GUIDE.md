# 🚀 Deployment Guide - Spinal Injury Detection System

## ⚠️ Important Understanding

**GitHub Repository ≠ Live Website**

When you visit `https://github.com/pronad1/Deploy-Model`, you see the **code repository** (README file).  
To make your Flask app accessible online, you need to **deploy it to a hosting platform**.

---

## 🌐 Deployment Options

### Option 1: Render.com (Recommended - Free & Easy)

✅ **Free tier available**  
✅ **Automatic deployments from GitHub**  
✅ **No credit card required**  
✅ **Handles large files with Git LFS**

#### Steps:

1. **Go to [Render.com](https://render.com)** and sign up with GitHub

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `pronad1/Deploy-Model`
   - Configure:
     ```
     Name: spinal-injury-detection
     Environment: Python 3
     Build Command: pip install -r requirements.txt
     Start Command: gunicorn app:app
     ```

3. **Add Environment Variables** (if needed)
   ```
   FLASK_ENV=production
   ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Your app will be live at: `https://spinal-injury-detection.onrender.com`

#### ⚠️ Render Free Tier Limitations:
- App sleeps after 15 min of inactivity (first request takes ~30 seconds)
- 512MB RAM limit (may need to optimize model loading)
- 750 hours/month free

---

### Option 2: Railway.app (Easy with Better Free Tier)

✅ **$5 free credit monthly**  
✅ **No sleep/cold starts**  
✅ **GitHub integration**

#### Steps:

1. **Go to [Railway.app](https://railway.app)** and sign up

2. **New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `pronad1/Deploy-Model`

3. **Configure**
   - Railway auto-detects Python
   - Add start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

4. **Deploy**
   - Click "Deploy"
   - Get URL from settings

---

### Option 3: Hugging Face Spaces (Best for ML Apps)

✅ **Free for public models**  
✅ **Designed for ML applications**  
✅ **Git LFS support built-in**

#### Steps:

1. **Create account at [Hugging Face](https://huggingface.co)**

2. **Create New Space**
   - Click "New Space"
   - Choose "Gradio" or "Streamlit" SDK
   - OR use Docker SDK

3. **Push Your Code**
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/spinal-detection
   git push hf main
   ```

4. **Access**: `https://huggingface.co/spaces/YOUR_USERNAME/spinal-detection`

---

### Option 4: Heroku (Requires Credit Card)

⚠️ **No free tier anymore (starts at $7/month)**

#### Steps:

1. **Install Heroku CLI**
   ```bash
   # Windows
   winget install Heroku.HerokuCLI
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create spinal-injury-app
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

4. **Open App**
   ```bash
   heroku open
   ```

---

### Option 5: Google Cloud Run (Pay-as-you-go)

✅ **Free tier: 2 million requests/month**  
✅ **Auto-scaling**  
✅ **Docker-based**

#### Steps:

1. **Install Google Cloud CLI**

2. **Build and Deploy**
   ```bash
   gcloud run deploy spinal-detection \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

---

## 🔧 Required Files (Already in Your Repo)

✅ **requirements.txt** - Python dependencies  
✅ **Procfile** - Heroku deployment config  
✅ **Dockerfile** - Docker deployment  
✅ **runtime.txt** - Python version  
✅ **.gitattributes** - Git LFS configuration  

---

## ⚠️ Important: Model File Size Issues

Your model files (177 MB total) are tracked with **Git LFS**. Some platforms may have issues:

### Solutions:

1. **Use Render/Railway** - They support Git LFS
2. **External Storage** - Move models to AWS S3/Azure Blob
3. **Model Compression** - Quantize models to reduce size

---

## 🎯 Quick Deployment (Render - Recommended)

```bash
# 1. Ensure code is pushed to GitHub
git add .
git commit -m "Ready for deployment"
git push origin main

# 2. Go to render.com
# 3. Connect GitHub repo
# 4. Use these settings:
#    Build: pip install -r requirements.txt
#    Start: gunicorn app:app
# 5. Deploy!
```

**Your app will be live in 5-10 minutes!**

---

## 📊 Testing Your Deployed App

Once deployed, test with:

1. **Health Check**: `https://your-app-url.com/health`
2. **Upload DICOM**: Use the web interface
3. **Monitor Logs**: Check platform logs for errors

---

## 🐛 Common Deployment Issues

### Issue 1: "Application Error" or 500 Error
**Solution**: Check logs for model loading errors. May need more RAM.

### Issue 2: "Slug size too large" (Heroku)
**Solution**: Use .slugignore to exclude unnecessary files:
```
*.ipynb
*.pdf
detection output/yolo11/runs/
```

### Issue 3: "Out of memory"
**Solution**: 
- Use smaller model versions
- Implement lazy loading
- Upgrade to paid tier with more RAM

### Issue 4: Cold Start (Render Free Tier)
**Solution**: Accept 30-second first load, or upgrade to paid tier

---

## 💡 Recommended Path for Your Project

**For FREE deployment:**
1. ✅ **Try Render.com first** (easiest)
2. ✅ **If Render fails, try Railway.app**
3. ✅ **If models too large, try Hugging Face Spaces**

**For PRODUCTION:**
1. 🚀 **AWS/Google Cloud** (scalable, reliable)
2. 🚀 **Azure** (good ML support)

---

## 📞 Need Help?

- **Render Docs**: https://render.com/docs
- **Railway Docs**: https://docs.railway.app
- **Hugging Face**: https://huggingface.co/docs/hub

---

## 🎉 After Deployment

Update your README.md with the live URL:
```markdown
🌐 **[Live Demo](https://your-app-name.onrender.com)**
```

Share your deployed app! 🚀
