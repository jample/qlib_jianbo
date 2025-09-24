from flask import Flask, render_template_string
import os

app = Flask(__name__)

# HTML template for the welcome page
WELCOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to Qlib Demo</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 3rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 600px;
            margin: 2rem;
        }
        h1 {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 1rem;
            font-weight: 300;
        }
        .subtitle {
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            line-height: 1.6;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        .feature {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .feature h3 {
            color: #333;
            margin: 0 0 0.5rem 0;
            font-size: 1.1rem;
        }
        .feature p {
            color: #666;
            margin: 0;
            font-size: 0.9rem;
        }
        .footer {
            margin-top: 2rem;
            color: #999;
            font-size: 0.9rem;
        }
        .status {
            background: #e8f5e8;
            color: #2d5a2d;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            display: inline-block;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="status">🚀 Server Running on Port 6006</div>
        <h1>Welcome to Qlib</h1>
        <p class="subtitle">
            A Python-based open source quantitative investment platform aimed at realizing the potential, 
            empowering the research, and creating the value of AI technologies in quantitative investment.
        </p>
        
        <div class="features">
            <div class="feature">
                <h3>📊 Data Management</h3>
                <p>Comprehensive data collection and processing capabilities for financial markets</p>
            </div>
            <div class="feature">
                <h3>🤖 AI Models</h3>
                <p>State-of-the-art machine learning models for quantitative trading strategies</p>
            </div>
            <div class="feature">
                <h3>📈 Backtesting</h3>
                <p>Robust backtesting framework to validate trading strategies</p>
            </div>
            <div class="feature">
                <h3>🔄 Workflow</h3>
                <p>End-to-end workflow management for quantitative investment research</p>
            </div>
        </div>
        
        <div class="footer">
            <p>Qlib Demo Server • Version {{ version }} • Running on Flask</p>
            <p>Visit <a href="https://github.com/microsoft/qlib" target="_blank">GitHub Repository</a> for more information</p>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def welcome():
    """Main welcome page"""
    return render_template_string(WELCOME_TEMPLATE, version="1.0.0")

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Qlib Demo Server",
        "port": 6006,
        "message": "Server is running successfully"
    }

@app.route('/about')
def about():
    """About page with system information"""
    return {
        "name": "Qlib Demo Server",
        "version": "1.0.0",
        "description": "A quantitative investment platform demo",
        "port": 6006,
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Welcome page"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/about", "method": "GET", "description": "About information"}
        ]
    }

if __name__ == '__main__':
    print("🚀 Starting Qlib Demo Server...")
    print("📍 Access the welcome page at: http://localhost:6006")
    print("💡 Health check available at: http://localhost:6006/health")
    print("ℹ️  About page available at: http://localhost:6006/about")
    print("🛑 Press Ctrl+C to stop the server")
    
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=6006,
        debug=True,
        use_reloader=True
    )