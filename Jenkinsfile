pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/HamzaKhan11111/Multimodal-neural-network-architecture-for-sentiment-analysis'
            }
        }

        stage('Build Image') {
            steps {
                bat 'docker build -t doc_image .'
            }
        }

        stage('Push to Docker Hub') {
            steps {
                bat 'docker login'
                bat 'docker tag doc_image hamzak11111/mlops_assignment_1:first_tag'
                bat 'docker push hamzak11111/mlops_assignment_1:first_tag'
            }
        }
    }
}