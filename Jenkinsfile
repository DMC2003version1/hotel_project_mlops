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

        stage("Building & Pushing Docker Image to Google Cloud Registry") {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo "Building & Pushing Docker Image to GCR"
                        sh '''
                            # ====== Export credentials ======
                            export GOOGLE_APPLICATION_CREDENTIALS="$GOOGLE_APPLICATION_CREDENTIALS"

                            # ====== Ensure gcloud and docker-credential-gcloud in PATH ======
                            export PATH="/usr/lib/google-cloud-sdk/bin:$PATH"

                            # ====== Check gcloud ======
                            echo "Using gcloud at: $(which gcloud)"
                            gcloud --version

                            # ====== Authenticate ======
                            gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
                            gcloud config set project "${GCP_PROJECT}"
                            gcloud auth configure-docker --quiet

                            # ====== Build & Push Docker Image ======
                            docker build -t gcr.io/${GCP_PROJECT}/hotel-project-mlops:latest .
                            docker push gcr.io/${GCP_PROJECT}/hotel-project-mlops:latest
                        '''
                    }
                }
            }
        }


        stage("Run Training Inside Container") {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GCP_KEY')]) {
                    sh """
                        docker run \
                            -e GOOGLE_APPLICATION_CREDENTIALS=/key.json \
                            -v $GCP_KEY:/key.json \
                            ${IMAGE_NAME} \
                            python pipeline/training_pipeline.py
                    """
                }
            }
        }
    }
}