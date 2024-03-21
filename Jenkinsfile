pipeline {
    agent any

    environment {
        DOCKER_HUB_CREDENTIALS = credentials('docker-hub-credentials')
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/HamzaKhan11111/Multimodal-neural-network-architecture-for-sentiment-analysis'
            }
        }

        stage('Containerize and Push to Docker Hub') {
            steps {
                script {
                    sh 'docker build -t doc_image .'
                }

                script {
                    docker.withRegistry('https://index.docker.io/v1/', DOCKER_HUB_CREDENTIALS)
                    {
                        sh 'docker push doc_image'
                    }
                }
            }
        }
    }
}
