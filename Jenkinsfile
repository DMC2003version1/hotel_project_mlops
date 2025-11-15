pipeline  {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage("Install Python") {
            steps{
                sh """
                    apt-get update
                    apt-get install -y python3 python3-venv python3-pip
                """
            }
        }
        
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
                        source ${VENV_DIR}/bin/activate
                        
                        pip install --upgrade pip
                        pip install -e .
                    """
                }
            }
        }
    }
}