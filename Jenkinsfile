pipeline {
    agent any
    
    environment {
        IMAGE_NAME = "cyber-def25-detector"
    }
    
    stages {
        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                script {
                    sh 'docker build -t $IMAGE_NAME .'
                }
            }
        }
        
        stage('Run with Docker Compose') {
            steps {
                echo 'Running malware detection with Docker Compose...'
                script {
                    sh 'docker-compose down || true'
                    sh 'docker-compose up -d'
                    sh 'sleep 10'
                    sh 'docker-compose ps'
                    sh 'cat output/alerts.csv || echo "Processing..."'
                }
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up...'
            sh 'docker-compose down || true'
            sh 'docker system prune -af --volumes || true'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs.'
        }
    }
}
```

---

## File 8: `.gitignore`
```
__pycache__/
*.pyc
output/
*.log
.vscode/
.idea/
.DS_Store
