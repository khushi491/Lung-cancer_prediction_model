# Railway Deployment Guide

## 🚂 Deploying Your Lung Cancer Risk Assessment API to Railway

### Prerequisites
- GitHub account with your code repository
- Railway account (free at [railway.app](https://railway.app))

### Step-by-Step Deployment

#### 1. Prepare Your Repository
Your repository is now ready for Railway deployment with the following files:
- `railway.json` - Railway configuration
- `Procfile` - Process definition
- `requirements.txt` - Python dependencies
- `app.py` - Updated Flask application
- `Dockerfile` - Alternative deployment option

#### 2. Deploy to Railway

1. **Go to Railway Dashboard**
   - Visit [railway.app](https://railway.app)
   - Sign in with your GitHub account

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Deployment**
   - Railway will automatically detect it's a Python project
   - The build process will use the standard Python builder
   - No additional configuration needed

4. **Deploy**
   - Click "Deploy Now"
   - Railway will build and deploy your application

#### 3. Access Your Application

Once deployed:
- Railway will provide a public URL (e.g., `https://your-app-name.railway.app`)
- Your API will be accessible at this URL
- The web interface will be available at the root path `/`
- The prediction API will be at `/predict`

### Key Features Added for Railway

#### 🔧 Production Optimizations
- **Gunicorn**: Production-grade WSGI server
- **CORS Support**: Cross-origin resource sharing enabled
- **Health Check**: `/health` endpoint for monitoring
- **Error Logging**: Comprehensive error handling and logging
- **Environment Variables**: Proper port configuration

#### 📊 Monitoring
- Health check endpoint: `GET /health`
- Returns application status and model loading status
- Railway will use this for health monitoring

#### 🔒 Security
- Debug mode disabled in production
- Proper error handling without exposing sensitive information
- CORS configured for web access

### Testing Your Deployment

#### 1. Test the Web Interface
```bash
curl https://your-app-name.railway.app/
```

#### 2. Test the Health Endpoint
```bash
curl https://your-app-name.railway.app/health
```

#### 3. Test the Prediction API
```bash
curl -X POST https://your-app-name.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "M",
    "age": 45,
    "smoking": 1,
    "yellow_fingers": 0,
    "anxiety": 1,
    "peer_pressure": 0,
    "chronic_disease": 0,
    "fatigue": 1,
    "allergy": 0,
    "wheezing": 1,
    "alcohol_consuming": 0,
    "coughing": 1,
    "shortness_of_breath": 0,
    "swallowing_difficulty": 0,
    "chest_pain": 0
  }'
```

### Railway-Specific Benefits

#### 🚀 Performance
- Automatic scaling based on traffic
- Global CDN for fast access
- Optimized build process

#### 💰 Cost
- Free tier available
- Pay-as-you-go pricing
- No credit card required for basic usage

#### 🔧 Management
- Easy rollbacks
- Environment variable management
- Built-in logging and monitoring
- GitHub integration for automatic deployments

### Troubleshooting

#### Common Issues

1. **Build Fails**
   - Check Railway logs for specific errors
   - Ensure all dependencies are in `requirements.txt`
   - Verify Python version compatibility

2. **Model Not Loading**
   - Ensure `lung_cancer_prediction_model.joblib` is in the repository
   - Check file size limits (Railway supports up to 1GB)
   - Verify the model file path in `app.py`

3. **Application Not Starting**
   - Check the `/health` endpoint
   - Review Railway logs for startup errors
   - Verify the `Procfile` configuration

#### Getting Help
- Railway documentation: [docs.railway.app](https://docs.railway.app)
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- Check Railway dashboard logs for detailed error information

### Next Steps

After successful deployment:
1. Test all endpoints thoroughly
2. Set up custom domain (optional)
3. Configure environment variables if needed
4. Set up monitoring and alerts
5. Consider setting up automatic deployments from your main branch

Your Lung Cancer Risk Assessment API is now ready for production use on Railway! 🎉 