pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/HamzaKhan11111/Multimodal-neural-network-architecture-for-sentiment-analysis'
            }
        }

        stage('Containerize and Push to Docker Hub') {
            steps {
                bat 'docker build -t doc_image .'
            }
        }

        stage('Push to Docker Hub') {
            steps {
                bat 'docker push doc_image'
            }
        }
    }
}