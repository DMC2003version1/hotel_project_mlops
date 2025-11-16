pipeline  {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "sodium-ray-476215-f3"
        GCLOUD_PATH = "/usr/lib/google-cloud-sdk/bin"
    }

    stages{
        stage("Clone Repository") {
            steps{
                script{
                    echo "Cloning Repository to Jenkins Workspace"
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/DMC2003version1/hotel_project_mlops.git']])
                }
            }
        }

        stage("Setting up Virtual Environment and Installing Dependencies") {
            steps{
                script{
                    echo "Setting up Virtual Environment and Installing Dependencies"
                    sh """
                        python -m venv ${VENV_DIR}
                        . ${VENV_DIR}/bin/activate
                        
                        pip install --upgrade pip
                        pip install -e .
                    """
                }
            }
        }

        stage("Building pushing Docker Image to Google Cloud Registry") {
            steps{
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo "Building pushing Docker Image to Google Cloud Registry"
                        sh """
                            export PATH=${GCLOUD_PATH}:${PATH}

                            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                            gcloud config set project ${GCP_PROJECT}

                            gcloud auth configure-docker --quiet

                            docker build -t gcr.io/${GCP_PROJECT}/hotel-project-mlops:latest .

                            docker push gcr.io/${GCP_PROJECT}/hotel-project-mlops:latest
                        """
                    }
                }

            }
        }
    }
}