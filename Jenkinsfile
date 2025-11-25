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
                    // Stop any existing containers
                    sh 'docker-compose down || true'
                    
                    // Start the application
                    sh 'docker-compose up -d'
                    
                    // Wait for processing
                    sh 'sleep 10'
                    
                    // Show container status
                    sh 'docker-compose ps'
                    
                    // Display results
                    sh 'cat output/alerts.csv || echo "Alerts file not ready yet"'
                }
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up...'
            sh 'docker-compose down || true'
            sh 'docker system prune -af --volumes'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for details.'
        }
    }
}
